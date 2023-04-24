import numpy as np
import torch

from config import *
from logger import Writer, log_line_to_dict, check_dict_for_attributes
from classic_training import train
import time
from data import *
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import glob
import os
import cost_modifying_functions
from utils import *

# Main experiment entry point

def get_optimizer(opt_name, qnn, lr):
    if opt_name.lower() == 'adam':
        optimizer = torch.optim.Adam
    elif opt_name.lower() == 'LBFGS':
        optimizer = torch.optim.LBFGS
    elif opt_name.lower() == 'RAdam':
        optimizer = torch.optim.RAdam
    elif opt_name.lower() == 'NAdam':
        optimizer = torch.optim.NAdam
    elif opt_name.lower() == 'ASGD':
        optimizer = torch.optim.ASGD
    elif opt_name.lower() == 'SparseAdam':
        optimizer = torch.optim.SparseAdam
    else:
        optimizer = torch.optim.SGD

    if isinstance(qnn.params, list):
        optimizer = optimizer(qnn.params, lr=lr)
    else:
        optimizer = optimizer([qnn.params], lr=lr)

    return optimizer


def get_scheduler(use_scheduler, optimizer,factor=0.8, patience=3, verbose=False):
    if use_scheduler:
        # some old values: factor=0.8, patience=10, min_lr=1e-10, verbose=False
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=1e-10, verbose=verbose)
    else:
        scheduler = None

    return scheduler


def process_execution(args):
    (process_id, num_processes, writer, idx_file_path, exp_file_path,
     x_qbits, cheat, qnn_name, lr, device, num_layers, opt_name, use_scheduler, num_epochs,
     scheduler_factor, scheduler_patience, small_std, data_generation) = args

    """Executes experiment
    For parameters see "exp".
    """

    print(f'Process with id {process_id}')
    if not exists(idx_file_path):
        raise ValueError('idx_file does not exist')
    try:
        current_idx = process_id - num_processes
        with open(idx_file_path, 'r') as idx_file:
            first_line = idx_file.readline().replace('\n', '')
            current_idx = int(first_line)
            idx_file.close()
        current_idx += num_processes

        line_dict = None
        attributes = dict(schmidt_rank='*', num_points='*', std='*', cost_modification='*')
        while True:
            with open(exp_file_path, 'r') as exp_file:
                current_line = exp_file.readline()
                for i in range(current_idx-1):
                    current_line = exp_file.readline()
                line_dict = log_line_to_dict(current_line)
                exp_file.close()

            if line_dict is None or not check_dict_for_attributes(line_dict, attributes):
                return

            # Get parameters for Experiment
            schmidt_rank = line_dict['schmidt_rank']
            num_points = line_dict['num_points']
            std = line_dict['std']
            cost_modification = line_dict['cost_modification']
            if cost_modification is not None:
                cost_modification = getattr(cost_modifying_functions, line_dict['cost_modification'])

            # Do experiment
            r_qbits = int(np.ceil(np.log2(schmidt_rank)))
            x_wires = list(range(x_qbits))

            if cheat:
                # Warning: Might not be fully supported 
                U, unitary_qnn_params = create_unitary_from_circuit(qnn_name, x_wires, cheat, device='cpu')
            else:
                U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)
            print(f"Run: current_exp_idx={current_idx}")
            info_string = f"schmidt_rank={schmidt_rank}, num_points={num_points}"

            # Generate Ansatz
            qnn = get_qnn(qnn_name, x_wires, num_layers, device=device)

            
            optimizer = get_optimizer(opt_name, qnn, lr)
            scheduler = get_scheduler(use_scheduler, optimizer, factor=scheduler_factor, patience=scheduler_patience)

            # Select type of training data
            if data_generation == "standard":
                # Standard training data (fully random for fixed rank)
                X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
            elif data_generation == "ortho":
                # Orthogonal training data for fixed rank
                X = torch.from_numpy(np.array(uniformly_sample_orthogonal_points(schmidt_rank, num_points, x_qbits, r_qbits)))
            elif data_generation == "nlihx":
                # Not linearly independent data for fixed rank
                r_qbits = x_qbits # for this type of data generation we keep the refsystem maximal
                while True:
                    # just to make sure: We test each dataset if it fulfills the requirements (should not fail with high probability)
                    X = np.array(sample_non_lihx_points(schmidt_rank, num_points, x_qbits, r_qbits))
                    is_ok, reason = check_non_lihx_points(X, schmidt_rank, x_qbits, r_qbits)
                    if not is_ok:
                        print("NLIHX creation failed because: %s - retrying" % reason)
                    else: 
                        # to torch
                        X = torch.from_numpy(X)
                        break
            elif data_generation == "avg_rank":
                # Average rank = schmidt_rank (other than that random)
                r_qbits = x_qbits # for this type of data generation we keep the refsystem maximal
                X = np.array(uniform_random_data_average(schmidt_rank, num_points, x_qbits, r_qbits))
                X = torch.from_numpy(X)
            elif data_generation == "avg_rank_evenly":
                # Generate training data evenly to ensure exact average rank for small datasets
                # Average rank = schmidt_rank (other than that random)
                r_qbits = x_qbits # for this type of data generation we keep the refsystem maximal
                X = np.array(uniform_random_data_average_evenly(schmidt_rank, num_points, x_qbits, r_qbits))
                
                nli = num_lin_ind(*X)
                nlihx = num_li_hx(X, 2**x_qbits, 2**r_qbits)
                ranks = [get_schmidt_rank(sample, 2**x_qbits, 2**r_qbits) for sample in X]
                mrank = np.mean(ranks)
                allnonortho = all_non_ortho(*X)
                # Debug info
                print("Some info about data in exp_id %d: Ranks: %s, Mean Rank: %s, Num_Lin_ind_HXR: %s, Num_Lin_ind_HX: %s, All Nonortho: %s"
                    % (current_idx, str(ranks), str(mrank), str(nli), str(nlihx), str(allnonortho)))
                
                X = torch.from_numpy(X)


            # Transform to "all in one" representation
            X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)

            # Start experiment
            starting_time = time.time()
            losses = train(X, U, qnn, num_epochs, optimizer, scheduler, cost_modification=cost_modification)
            train_time = time.time() - starting_time
            print(f"\tTraining took {train_time}s")

            # Compute risk
            risk = quantum_risk(U, qnn.get_matrix_V())
            print("Final RISK %s in exp id %d" % (str(risk), current_idx))

            # Log everything
            if writer:
                losses_str = str(losses).replace(' ', '')
                qnn_params_str = str(qnn.params.tolist()).replace(' ', '')
                u_str = str(qnn.params.tolist()).replace(' ', '')
                writer.append_line(
                    info_string + f", std={0}, losses={losses_str}, risk={risk}, train_time={train_time}, qnn={qnn_params_str}, unitary={u_str}"
                )

            current_idx += num_processes
            with open(idx_file_path, 'w') as idx_file:
                idx_file.write(str(current_idx))
                idx_file.close()
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        raise e


def generate_exp_data(x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name,
                       device, cheat, use_scheduler, opt_name, scheduler_factor=0.8, scheduler_patience=3, std=False,
                       writer_path=None, num_processes=1, run_type='new', small_std=False,
                      schmidt_ranks=None, num_datapoints=None, std_list=None, cost_modification="identity", data_generation="standard"):
    """
    Generate experiment data and spawn processes with experiments - for parameters see "exp"
    """
    #ProcessPoolExecutor
    idx_file_dir = writer_path
    if run_type != 'continue':
        filelist = glob.glob(os.path.join(idx_file_dir, "process_idx_*.txt"))
        for f in filelist:
            os.remove(f)


    writers = [None]*num_processes
    if writer_path:
        writers = [Writer(writer_path+f"result_{process_id}.txt", delete=(run_type != 'continue')) for process_id in range(num_processes)]
        for writer in writers:
            writer.append_line(f"x_qbits={x_qbits}, num_layers={num_layers}, num_epochs={num_epochs}, lr={lr}, "
                          f"num_unitaries={num_unitaries}, num_datasets={num_datasets}, qnn_name={qnn_name}, "
                          f"device={device}, cheat={cheat}, use_scheduler={use_scheduler}")

    exp_file_path = gen_exp_file(x_qbits, num_unitaries, num_datasets, std, small_std, schmidt_ranks, num_datapoints, std_list, cost_modification)
    complete_starting_time = time.time()

    # (process_id, num_processes, writer, idx_file_path, exp_file_path,
    #  x_qbits, cheat, qnn_name, device, num_layers, opt_name, use_scheduler, num_epochs)

    ppe = ProcessPoolExecutor(max_workers=num_processes)
    worker_args = []
    for process_id in range(num_processes):
        idx_file_path = idx_file_dir + f'process_idx_{process_id}.txt'
        if not exists(idx_file_path):
            with open(idx_file_path, 'w') as idx_file:
                idx_file.write(str(process_id-num_processes))
                idx_file.close()
        worker_args.append((process_id, num_processes, writers[process_id], idx_file_path, exp_file_path, x_qbits,
                            cheat, qnn_name, lr, device, num_layers, opt_name, use_scheduler, num_epochs,
                            scheduler_factor, scheduler_patience, small_std, data_generation))
    results = ppe.map(process_execution, worker_args)
    for res in results:
        print(type(res))
        print(dir(res))
        print(res)
    # iterate over untrained unitaries
    zero_risks = []
    for unitary_idx in range(num_unitaries):
        x_wires = list(range(x_qbits))
        if cheat:
            U, _ = create_unitary_from_circuit(qnn_name, x_wires, cheat, device='cpu')
        else:
            U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)
        for dataset_idx in range(num_datasets):
            qnn = get_qnn(qnn_name, x_wires, num_layers, device='cpu')
            zero_risks.append(quantum_risk(U, qnn.get_matrix_V()))
    zero_risks = np.array(zero_risks)
    if writers:
        writers[0].append_line(f"zero_risks={zero_risks}")

    complete_time = time.time()-complete_starting_time
    print(f"Complete experiment took {complete_time}s")
    for writer in writers:
        writer.append_line(f"complete_time={complete_time}")


def exp(x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name, device, cheat, use_scheduler,
        optimizer, scheduler_factor=0.8, scheduler_patience=3, std=False, writer_path=None, num_processes=1, run_type='continue',
        small_std=False,
        schmidt_ranks=None, num_datapoints=None, std_list=None, cost_modification="identity", data_generation="standard"):
    """
    Main entry point for experiments
    x_qbits: Number of qubits for H_X system
    num_layers: Number of layers in ansatz
    num_epochs: Number of epochs for training
    lr: Learning rate
    num_unitaries: Number of unitaries to traing
    num_datasets: Number of datasets to traing
    qnn_name: Type of ansatz to use
    device: Device for pytorch computation
    cheat: Not fully implemented - train using unitaries generated from a qnn instead of random
    use_scheduler: Allow sceduler in training
    optimizer: Optimizer for training
    scheduler_factor, scheduler_patience: Scheduler settings
    (old/unused) std: Deviation for schmidt ranks
    writer_path: Output directory
    num_processes: Number of processes to spawn with experiment
    (old/unused) run_type:
    (old/unused) small_std: Deviation for schmidt ranks
    schmidt_ranks: Schmidt rank to use
    num_datapoints: Number of samples
    (old/unused) std_list: List of deviations
    (old/unused) cost_modification: Modification of cost function
    data_generation: Method for training data generation
    """

    generate_exp_data(
        x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name, device, cheat, use_scheduler,
        optimizer, scheduler_factor=scheduler_factor, scheduler_patience=scheduler_patience, std=std,
        writer_path=writer_path, num_processes=num_processes, run_type=run_type, small_std=small_std,
        schmidt_ranks=schmidt_ranks, num_datapoints=num_datapoints, std_list=std_list, cost_modification=cost_modification,
        data_generation=data_generation
    )
