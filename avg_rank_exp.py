from generate_experiments import *
from pathlib import Path
import sys

# Experiments for training QNNs with varying Schmidt rank data
# Setup:
# target unitary U randomly generated with 6 qubits
# training data randomly generated using data.uniform_random_data_average_evenly
# 200 combinations of unitary + training data
# trains using multiprocessing: 1 experiment per core for num_processes cores (see below)
# creates directory experimental_results/avg_rank_data/t[a]r[b] for experiments with "a" samples and rank "b"

# other parameters:
# n_datapoints: numbers of training samples (t)
# schmidt_rank: schmidt_ranks for the experiments
# lr: learning rate for optimization
# scheduler_*: Scheduler settings
# n_layers: number of ansatz layers (400 works good for 6 qubits)


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')

    x_qubits = 6 
    n_layers = 400 
    num_datasets_per_unitary = 20
    num_unitaries = 10
    scheduler_factor = 0.8
    scheduler_patience = 10
    num_processes = 24
    lr = 0.1
    run_type = 'new'
    n_datapoints = [1,2,4,8,16,32,64]
    print("total set of training data sizes: ", n_datapoints)
    sys.stdout.flush()

    for t in n_datapoints:
        schmidt_rank = [64,4,2,1] 
        for rank in schmidt_rank: # starting them separately so results are obtained by t
        
            num_datapoints = [t]
            std_list = [0]

            print("Starting experiments with t=%d datasets of rank r=%d" % (t, rank))
            sys.stdout.flush()

            schmidt_ranks = [int(rank)]
            writer_path = './experimental_results/avg_rank_data/t' + str(t) + 'r' + str(rank) + '/'
            Path(writer_path).mkdir(parents=True, exist_ok=True)

            exp(x_qubits, n_layers, 1000, lr, num_unitaries, num_datasets_per_unitary, 'CudaPennylane', 'cpu', None, True, 'Adam',
                scheduler_factor=scheduler_factor, scheduler_patience=scheduler_patience, std=False,
                writer_path=writer_path, num_processes=num_processes, run_type=run_type,
                small_std=False,
                schmidt_ranks=schmidt_ranks,
                num_datapoints=num_datapoints,
                std_list=None,
                cost_modification="identity",
                data_generation="avg_rank_evenly"
                )

