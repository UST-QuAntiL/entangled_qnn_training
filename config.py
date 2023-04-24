from os.path import exists
from logger import Writer


def gen_config(rank, num_points, x_qbits, r_qbits, num_unitaries, num_layers, num_training_data,
               mean, std = 0, learning_rate=0.01, batch_size=8, num_epochs=120, shuffle=True, optimizer='COBYLA'):
    '''
    schmidt_rank -- schmidt rank of states in training data (see param std)
    num_points -- number of points in each training dataset
    x_qbits -- number of input qubits
    r_qbits -- number of qubits in the reference system
    num_unitaries -- number of unitaries to be generated
    num_layers -- number of layers in NN architecture
    num_train_data -- number of training datasets
    std -- std deviation of schmidt rank (use schmidt_rank for all samples if this is 0)
    '''
    config = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        schmidt_rank=rank,
        num_points=num_points,
        x_qbits=x_qbits,
        r_qbits=r_qbits,
        num_unitaries=num_unitaries,
        num_layers=num_layers,
        num_training_data=num_training_data,
        mean=mean,
        std=std,
        optimizer=optimizer
    )
    return config

def gen_exp_file(x_qbits, num_unitaries, num_datasets, std_bool=False, small_std=False, schmidt_ranks=None, num_datapoints=None, std_list=None, cost_modification="identity"):
    file_path = f'./data/{x_qbits}_exp_file.txt'
    writer = Writer(file_path)
    if not small_std:
        if schmidt_ranks is None:
            schmidt_ranks = [2**i for i in range(x_qbits+1)]
        if num_datapoints is None:
            num_datapoints = list(range(1, 2 ** x_qbits + 1))
        for schmidt_rank in schmidt_ranks:
            for num_points in num_datapoints:
                for unitary_idx in range(num_unitaries):
                    for dataset_idx in range(num_datasets):
                        if not std_bool:
                            writer.append_line(f"schmidt_rank={schmidt_rank}, num_points={num_points}, std=0, "
                                               f"unitary_idx={unitary_idx}, dataset_idx={dataset_idx}, cost_modification={cost_modification}")
                        else:
                            max_rank = 2 ** x_qbits
                            if not std_list:
                                std_list = range(1, max_rank)
                            for std in std_list:
                                if min(schmidt_rank - 1, max_rank - schmidt_rank) < 3 * std:
                                    continue
                                writer.append_line(f"schmidt_rank={schmidt_rank}, num_points={num_points}, std={std}, "
                                                   f"unitary_idx={unitary_idx}, dataset_idx={dataset_idx}, cost_modification={cost_modification}")

    else:
        if schmidt_ranks is None:
            schmidt_ranks = [4]
        if num_datapoints is None:
            num_datapoints = [2]
        for schmidt_rank in schmidt_ranks:
            for num_points in num_datapoints:
                max_std = min(2**x_qbits - schmidt_rank, schmidt_rank - 1) + 1
                if std_list is None:
                    temp_std_list = list(range(max_std))
                else:
                    temp_std_list = []
                    for std in std_list:
                        if std < max_std:
                            temp_std_list.append(std)
                for unitary_idx in range(num_unitaries):
                    for dataset_idx in range(num_datasets):
                        for std in temp_std_list:
                            writer.append_line(f"schmidt_rank={schmidt_rank}, num_points={num_points}, std={std}, "
                                               f"unitary_idx={unitary_idx}, dataset_idx={dataset_idx}, cost_modification={cost_modification}")
    return file_path
