import pennylane as qml
import numpy as np
import torch
import networkx as nx

# Utility functions for QNN simulations

# tolerance for numpy =0 compares for orthonormality checks
tolerance = 1E-10

# Encodings

def int_to_bin(num, num_bits):
    """
    Convert integer to binary with padding
    (e.g. (num=7, num_bits = 5) -> 00111)
    """
    b = bin(num)[2:]
    return [0 for _ in range(num_bits - len(b))] + [int(el) for el in b]

def one_hot_encoding(num, num_bits):
    """
    Returns one-hot encoding of a number
    (e.g. (num=4, num_bits=7) -> 0000100)
    """
    result = [0]*num_bits
    result[num] = 1
    return result

def normalize(point):
    """Normalizes vector"""
    return point / np.linalg.norm(point)

def num_li_hx(vectors, dim_x, dim_r):
    """number of linear independent vectors in hx"""
    hx_vectors_total = []
    for vec in vectors:
        coeffs, lefts, rights = schmidt_decomp(vec, dim_r, dim_x)
        hx_vectors = np.array(rights)[np.array(coeffs) > 1E-10] # tolerance based
        for vec in hx_vectors:
            hx_vectors_total.append(vec)
    return num_lin_ind(hx_vectors_total)

def tensor_product(state1: np.ndarray, state2: np.ndarray):
    result = np.zeros(len(state1)*len(state2), dtype=np.complex128)
    for i in range(len(state1)):
        result[i*len(state2):i*len(state2)+len(state2)] = state1[i] * state2
    return result

def torch_tensor_product(matrix1: torch.Tensor, matrix2: torch.Tensor, device='cpu'):
    result = torch.zeros((matrix1.shape[0]*matrix2.shape[0], matrix1.shape[1]*matrix2.shape[1]), dtype=torch.complex128, device=device)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            result[i*matrix2.shape[0]:i*matrix2.shape[0]+matrix2.shape[0], j*matrix2.shape[1]:j*matrix2.shape[1]+matrix2.shape[1]] = matrix1[i, j] * matrix2
    return result

def adjoint_unitary_circuit(unitary):
    """
    Generates a circuit corresponding to the adjoint of the input matrix
    """
    from qiskit import QuantumCircuit, Aer, transpile

    unitary = np.conj(np.array(unitary)).T

    qbits = int(np.log2(len(unitary)))
    sv_backend = Aer.get_backend('statevector_simulator')

    qc = QuantumCircuit(qbits)
    qc.unitary(unitary, range(qbits))
    qc_transpiled = transpile(qc, backend=sv_backend, basis_gates=sv_backend.configuration().basis_gates,
                              optimization_level=3)
    return qml.from_qiskit(qc_transpiled)

# Data integrity checks

def all_ortho(*vecs):
    """All points orthogonal?"""
    for a in vecs:
        for b in vecs:
            product = np.abs(np.vdot(a, b))
            if np.any(a != b) and product > tolerance:
                #print("Non ortho: ")
                #print(product)
                #print(a)
                #print(b)
                return False
    return True


def all_non_ortho(*vecs):
    """All non-orthogonal?"""
    found_ortho = 0
    max_prod = 0
    for a in vecs:
        for b in vecs:
            product = np.abs(np.vdot(a, b))
            if np.any(a != b) and product > max_prod:
                max_prod = product
            if np.any(a != b) and product < tolerance:
                found_ortho += 1
    return found_ortho == 0

def num_ortho(*vecs):
    """Number of orthogonal pts"""
    found_ortho = 0
    max_prod = 0
    for a in vecs:
        for b in vecs:
            product = np.abs(np.vdot(a, b))
            if np.any(a != b) and product > max_prod:
                max_prod = product
            if np.any(a != b) and product < tolerance:
                found_ortho += 1
                
    return found_ortho/float(2)

def comp_basis(dim, i):
    """Computational basis vector with dimension 'dim'"""
    e = np.zeros((dim))
    e[i] = 1
    return e

def get_coeff(input, compbasis_entry):
    """Coeffiction for comp basis vector in state"""
    return np.vdot(compbasis_entry, input) 

def schmidt_decomp(v, dim_a, dim_b):
    """Schmidt decomposition by SVD using numpy"""
    # start with computational basis for H_A and H_B
    # therefore i can just read out the values of M
    M = np.zeros((dim_a, dim_b), dtype=np.complex128)
    for i in range(dim_a):
        e_i = comp_basis(dim_a, i)
        for j in range(dim_b):
            f_j = comp_basis(dim_b, j)
            M[i,j] = get_coeff(v, np.kron(e_i, f_j))

    U, sig, V = np.linalg.svd(M) 

    lefts = [0] * len(sig)
    rights = [0] * len(sig)
    coeffs = [0] * len(sig)
    for i in range(0, len(sig)):
        coeffs[i] = sig[i]
        lefts[i] = U[:, i] #U.col(i)
        rights[i] = V[i, :] # is already conj conjugate(V).col(i)

    return coeffs, lefts, rights

def get_schmidt_rank(v, dim_a, dim_b):
    """Computes schmidt rank using decomposition"""
    return len([coeff for coeff in schmidt_decomp(v, dim_a, dim_b)[0] if not np.isclose(coeff, 0)]) # extracts the length of the list of coefficients

def randnonzero():
    val = np.random.random()
    if val == 0: # if we really manage to get 0, we try again
        return randnonzero()
    else: 
        return val

def num_lin_ind(*vecs):
    """Number of linearly independent vectors"""
    M = np.row_stack(vecs)
    _, S, _ = np.linalg.svd(M)#
    # Tolerance:
    S[S < tolerance] = 0
    return np.count_nonzero(S)

def orthogonality_graph(*vecs):
    """Generates graph with edge for each non-orthogonal pair. 
    If graph not connected, non-orthogonality constraint is not satisfied."""
    # assigns an edge if 2 vectors are non-ortho
    g = nx.Graph()
    g.add_nodes_from(list(range(0, len(vecs))))
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if np.any(a != b):
                product = np.abs(np.vdot(a, b))
                if product > tolerance:
                    # non-ortho = assign edge
                    g.add_edge(i,j,weight = product)
    return g
