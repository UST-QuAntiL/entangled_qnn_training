from qnns.qnn import *
from data import *
from qnns.quantum_qnn import *
import classic_training
import quantum_training
import time


def quantum_risk(U, V):
    """
    Computes the quantum risk of a hypothesis unitary V
    with respect to the 'true' unitary U according to
    Equation A6 in Sharma et al.
    """
    dim = len(U)
    prod = torch.matmul(U.T.conj(), V)
    tr = abs(torch.trace(prod))**2
    risk = 1 - ((dim + tr)/(dim * (dim+1)))

    return risk


def calc_risk_qnn(trained_qnn, U):

    #circuit = construct_circuit(trained_params, num_layers, x_qbits)
    V = trained_qnn.get_matrix_V()
    risk = quantum_risk(U, V)
    return risk


def main():
    from config import gen_config
    gen_config()
    print(dict.values())
    print(dict.keys())
    # print(calc_avg_risk(**params))


if __name__ == '__main__':
    main()

