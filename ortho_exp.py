from generate_experiments import *
from pathlib import Path
import sys

# Experiments for training QNNs with orthogonal entangled training data
# Setup:
# target unitary U randomly generated with 6 qubits
# training data randomly generated using data.uniformly_sample_orthogonal_points
# 200 combinations of unitary + training data
# trains using multiprocessing: 1 experiment per core for num_processes cores (see below)
# creates directory experimental_results/nlihx_data/t[a] for experiments with "a" samples
# schmidt rank is fixed to r=d/t

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
    n_datapoints = [2**i for i in range(0, x_qubits+1)]
    print("total set of training data sizes: ", n_datapoints)
    sys.stdout.flush()

    for t in n_datapoints:
        schmidt_rank = (2**x_qubits) / t #=> r*t always matches the dimension
        
        num_datapoints = [t]
        std_list = [0]

        print("Starting experiments with t=%d datasets of rank r=%d" % (t, schmidt_rank))
        sys.stdout.flush()

        schmidt_ranks = [int(schmidt_rank)]
        writer_path = './experimental_results/ortho_data/t' + str(t) + '/'
        Path(writer_path).mkdir(parents=True, exist_ok=True)

        exp(x_qubits, n_layers, 1000, lr, num_unitaries, num_datasets_per_unitary, 'CudaPennylane', 'cpu', None, True, 'Adam',
            scheduler_factor=scheduler_factor, scheduler_patience=scheduler_patience, std=False,
            writer_path=writer_path, num_processes=num_processes, run_type=run_type,
            small_std=False,
            schmidt_ranks=schmidt_ranks,
            num_datapoints=num_datapoints,
            std_list=None,
            cost_modification="identity",
            data_generation="ortho"
            )

