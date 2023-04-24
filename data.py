from torch.utils.data import DataLoader
from scipy.stats import unitary_group
from utils import *
from typing import List
from qnns.qnn import get_qnn
import math
import copy
import random

# Training data Generation functions. 
# Used for Experiments:
# uniform_random_data_average_evenly: For data of average schmidt rank
# uniformly_sample_orthogonal_points: For orthogonal data
# sample_non_lihx_points: For linearly dependent data

def create_unitary_from_circuit(qnn_name, x_wires, num_layers, device='cpu'):
    """
    Randomly initializes given circuit with random chosen parameters
    and create unitary from intitialized circuit

    Parameters
    ----------
    qnn_name : str
        Name of the QNN to be used
    x_wires: list
        List of wires of our system
    num_layers: int
        Number of layers to use for QNN

    """
    unitary_qnn = get_qnn(qnn_name, x_wires, num_layers, device=device)
    if isinstance(unitary_qnn.params, list):
        unitary_qnn.params = [torch.tensor(np.random.normal(0, np.pi, unitary_qnn.params[i].shape), device=device) for i in range(len(unitary_qnn.params))]
    else:
        unitary_qnn.params = torch.tensor(np.random.normal(0, np.pi, unitary_qnn.params.shape), device=device)
    return unitary_qnn.get_tensor_V(), unitary_qnn.params


def uniformly_sample_from_base(num_qbits: int, size: int):
    """
    Draws vectors from an orthonormal basis (generated from the standard basis
    by multiplication with a random unitary)

    Parameters
    ----------
    num_qbits : int
        number of qubits of the input space -> dimension is 2**num_qbits
    size : int
        number of basis vectors to be drawn
    """
    if num_qbits == 0:
        return np.ones((1, 1))
    # uniform sampling of basis vectors
    num_bits = int(np.power(2, num_qbits))
    base = []
    random_ints = np.random.choice(num_bits, size, replace=False)
    transform_matrix = unitary_group.rvs(num_bits)
    for rd_int in range(len(random_ints)):
        binary_base = one_hot_encoding(random_ints[rd_int], num_bits)
        base.append(binary_base)

    return np.array(base) @ transform_matrix



def uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits):
    """
    Generates a random point with a specified schmidt rank by drawing basis vectors corresponding
    to the schmidt rank and 'pairing' them in a linear combination of elementary tensors

    Parameters
    ----------
    schmidt_rank : int
        determines how many basis vectors are drawn for the circuit and the reference system
    x_qbits, r_qbits: int
        specify the amount of qubits in the circuit and reference system
    """
    basis_x = uniformly_sample_from_base(x_qbits, schmidt_rank)
    basis_r = uniformly_sample_from_base(r_qbits, schmidt_rank)
    coeff = np.random.uniform(size=schmidt_rank)
    point = np.zeros((2**x_qbits * 2**r_qbits), dtype=np.complex128)
    for i in range(schmidt_rank):
        point += coeff[i] * tensor_product(basis_r[i], basis_x[i])
    return normalize(point)

def comp_basis(dim, i):
    e = np.zeros((dim))
    e[i] = 1
    return e

def randomized_bell_basis_state(n, m, d, r):
    """Generates bell basis state according to paper with randomized coefficient"""
    # reference system has dimension r, working system has dimension d
    # no check is made if those dimensions are powers of 2
    # dimension of reference system is the first power of two that can hold the integer r-1 
    # (which is the largest "tag" used in the reference system)
    # to match the training processes, the reference system is the first
    # (leftmost) system and the target system is second
    if r == 1:
        dim_r = 0
        point = np.zeros((d), dtype=np.complex128)
        # no entanglement => just sample from comp basis
        for k in range(0, r):
            coeff = np.random.random()
            vec = comp_basis(d, ((k+m)%d))
            point += coeff * np.exp((1j*2*np.pi/r)*n*k) * vec
        return normalize(point)
    else:
        qubits_r = int(np.floor(math.log(r-1,2))+1)
        dim_r = 2**qubits_r
        #print("Dim r: ", dim_r)
        point = np.zeros((d*dim_r), dtype=np.complex128)
        for k in range(0, r):
            coeff = np.random.random()
            vec = np.kron(comp_basis(dim_r, k), comp_basis(d, ((k+m)%d)))
            point += coeff * np.exp((1j*2*np.pi/r)*n*k) * vec
        return normalize(point)

def uniformly_sample_orthogonal_points(schmidt_rank: int, size: int, x_qubits: int, r_qubits: int, modify=True):
    """Generates a set of orthogonal points for learning. The points all have the given schmidt rank
    and are linearly independent in H_X.
    The points are sampled from a subset of the randomized bell states and modified with 
    uniformly ranodm W_X \otimes W_R. If modify = false, this step is omitted."""
    
    # get unitaries for X and R system
    x_transform = unitary_group.rvs(2**x_qubits)
    if r_qubits > 0:
        r_transform = unitary_group.rvs(2**r_qubits)
        combined_transform = np.kron(r_transform, x_transform)
    else:
        # no ref system
        combined_transform = x_transform

    # To obtain a proper set used for training we dont just sample from the set of all vectors, but 
    # we generated them with m = r*j for varying 0<=j<t. This way (from the construction of the vectors),
    # we will obtain a set that only uses orthogonal vectors in H_X
    # n is kept at 0
    comp_basis_states = [randomized_bell_basis_state(n, schmidt_rank*j, 2**x_qubits, schmidt_rank) for n in range(0, 1) for j in range(0, size)]
    # transform the states
    if modify:
        transformed_states = [combined_transform @ state for state in comp_basis_states]
        #print("transformed states shape", np.shape(transformed_states))
        return transformed_states
    else:
        return comp_basis_states

def uniform_random_data(schmidt_rank: int, size: int, x_qbits: int, r_qbits: int) -> List[List[float]]:
    """
    Generates a data set of specified size with a given schmidt rank by drawing points
    with uniformly_sample_random_point

    Parameters
    ----------
    schmidt_rank : int
        Desired Schmidt rank of the points in the data set
    size : int
        Desired size of the data set (number of points)
    x_qbits, r_qbits : int
        Desired input size of the circuit and reference system
    """
    data = []
    # size = number data samples of trainset
    for i in range(size):
        data.append(uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits))
    return data

def get_rank_range(R, dim):
    """returns maximal allowed range for schmidt ranks of inputs.
    allowed is everything 1 <= r <= dim. sets the range s.t. 
    the offset to each side of R is equal - therefore the mean should be R"""
    offset = min(R-1, dim-R)
    return (R-offset, R+offset)

def uniform_random_data_average(schmidt_rank: int, size: int, x_qbits: int, r_qbits: int):
    """same as uniform_random_data, however schmidt rank is only the 
    average rank R. r_qubits should equal x_qbits (not all necessarily 
    required but in the worst case)."""
    # allowed range
    rank_range = get_rank_range(schmidt_rank, 2**x_qbits)
    if r_qbits < x_qbits:
        raise "Reference system too small to hold maximally entangled sample."
    data = []
    # size = number data samples of trainset
    for i in range(size):
        # sample rank
        rank = np.random.randint(rank_range[0], rank_range[1]+1) #+1 since high is exclusive
        data.append(uniformly_sample_random_point(rank, x_qbits, r_qbits))
    return data

def uniform_random_data_average_evenly(schmidt_rank: int, size: int, x_qbits: int, r_qbits: int):
    """same as uniform_random_data_average, but the points are sampled evenly:
    That means for each point with rank R+k that is used, one point with 
    rank R-k is also created - this fixes the problem that the average 
    is not always correctly reached when training with uniformly sampled points 
    in the range for small number of t."""
    # allowed range
    rank_range_pos = get_rank_range(schmidt_rank, 2**x_qbits)[1] - schmidt_rank
    if r_qbits < x_qbits:
        raise "Reference system too small to hold maximally entangled sample."
    if size % 2 != 0:
        if size == 1:
            return [uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits)]
        else:
            raise "Can only generate inputs of even size t."
    data = []
    # size = number data samples of trainset
    for i in range(int(size/2)):
        # sample rank offset
        offset = np.random.randint(0, rank_range_pos+1) #+1 since high is exclusive
        data.append(uniformly_sample_random_point(schmidt_rank+offset, x_qbits, r_qbits))
        data.append(uniformly_sample_random_point(schmidt_rank-offset, x_qbits, r_qbits))
    return data


def create_pairs(mean, lower_bound, upper_bound, std):

    """
    Create pairs for std experiment 4-qubit system and rank x
    """
    return (mean-std, mean+std)


#create dataset of size <size> with a mean schmidt rank
def uniform_random_data_mean_pair(mean, std, num_samples, x_qbits):

    data = []
    lower_bound = 1
    upper_bound = 16
    tuple = create_pairs(mean, lower_bound, upper_bound, std)
    r_qbits = int(np.ceil(mean+std))

    if num_samples % 2 == 0:
        #no randomness
        for i in range(0, num_samples //2):
            data.append(uniformly_sample_random_point(mean-std, x_qbits, r_qbits))
        for i in range(0, num_samples//2):
            data.append(uniformly_sample_random_point(mean+std, x_qbits, r_qbits))

    else:

        #randomness to reduce bias that dataset are not of equal size
        # thorugh lots of iterations bias is reduced
        flag = np.random.randint(2, size = 1)
        for i in range(0, num_samples//2):
            data.append(uniformly_sample_random_point(mean - std, x_qbits, r_qbits))
        for i in range(0, num_samples//2):
            data.append(uniformly_sample_random_point(mean+std, x_qbits, r_qbits))

        if flag == 0:
            data.append(uniformly_sample_random_point(mean - std, x_qbits, r_qbits))
        else:
            data.append(uniformly_sample_random_point(mean + std, x_qbits, r_qbits))
    """"
        numbers_mean_std, counter, final_std, final_mean = create_mean_std(mean, std, num_samples)
        r_qbits = int(np.ceil(np.log2(numbers_mean_std.max())))
        for i in range(len(numbers_mean_std)):
            schmidt_rank = int(numbers_mean_std[i])
            data.append(uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits))
        """

    return data, std, mean

def generate_dependent_base(base_size, original_base):
    """Creates new base that is linearly dependent on original_base"""
    # pick two reals gamma, delta s.t. |gamma|^2 + |delta|^2 = 1
    gamma = randnonzero()
    delta = np.sqrt(1 - np.abs(gamma)**2)

    if len(original_base) == 1:
        # if we only have one lihx vector, all other linearly dep. 
        # vectors are multiples of this vector
        # since this is a global factor, the quantum states are 
        # the same, so just the same base is returned
        return original_base 

    # The new base is composed of two parts A and B
    # A part: for 2-subset of original base: gamma|a> + delta|b>
    # B part: for same 2-subsets: delta|a> - gamma|b>
    cbase = copy.deepcopy(original_base)
    newbase = []
    for _ in range(0, int(base_size/2)):
        # sample unitary 
        T = unitary_group.rvs(2)
        x_coeffs = T @ np.array([1,0])
        y_coeffs = T @ np.array([0,1])
        # pick and remove two elements
        elemA = cbase.pop(random.randint(0, len(cbase)-1))
        elemB = cbase.pop(random.randint(0, len(cbase)-1))

        # A part
        newbase.append(x_coeffs[0] * elemA + x_coeffs[1] * elemB)
        # B part
        newbase.append(y_coeffs[0] * elemA - y_coeffs[1] * elemB)

    return newbase

def generate_point_from_vecs(schmidt_rank, vecs_x, vecs_r, dim_x, dim_r):
    point = np.zeros((dim_x * dim_r), dtype=np.complex128)
    coeffs = normalize(np.random.random_sample(schmidt_rank))
    for k in range(0, schmidt_rank):
        point += coeffs[k] * np.kron(vecs_r[k], vecs_x[k])
    return point

def sample_non_lihx_points(schmidt_rank: int, size: int, x_qubits: int, r_qubits: int, modify=True):
    # generate the basis for the starting sample
    x_transform = unitary_group.rvs(2**x_qubits)
    base1 = [comp_basis(2**x_qubits, i) for i in range(0, schmidt_rank)]
    if modify:
        base1 = [x_transform @ state for state in base1]

    # now generate size-1 other bases that are composed of vectors that are linear dependent on vectors in base1
    bases = [base1]
    for i in range(0, size-1):
        newbase = generate_dependent_base(schmidt_rank, base1)
        #print("generated base: ", newbase)
        bases.append(newbase)


    # Now create the inputs by assigning vectors from a schmidt_rank * size element basis to the reference system
    refdim = schmidt_rank * size
    compb = [comp_basis(refdim, i) for i in range(0, refdim)]
    
    finalpoints = []
    for i in range(0, size):
        refamount = max(2, schmidt_rank) # amount of vectors in refsystem needed
        rr_transform = unitary_group.rvs(refdim)
        refbasis = [rr_transform @ compb[i] for i in range(0, refamount)]

        pt = generate_point_from_vecs(schmidt_rank, bases[i], refbasis, 2**x_qubits, refdim)# refdim)
        finalpoints.append(pt)

    return finalpoints


def check_non_lihx_points(points, schmidt_rank, x_qubits, r_qubits):
    # Checks for non lihx points of all properties are met:
    # they have to be linear dependent in hx (prop 2 not sat)
    # => num of lihx vectors = schmidt_rank
    # They have to be linear independent in hxr (prop 1 sat)
    # => num of li vectors = len(points)
    # They have to be non-orthogonal => othogonality graph is connected (prop 3 sat)

    # Prop 1
    if num_lin_ind(*points) != len(points):
        return False, "They are not lin ind. in H_XR"
    
    # Prop 2
    if num_li_hx(points, 2**x_qubits, 2**r_qubits) != schmidt_rank:
        return False, "They are not linear dependent in H_X"

    # Prop 3
    g = orthogonality_graph(*points)
    if not nx.is_connected(g):
        return False, "The non-ortho graph is not connected!"

    return True, "OK"

def random_unitary_matrix(x_qbits):
    """
    Generates Haar-distributed unitary

    Parameters
    ----------
    x_qbits : int
        Dimension of input system -> unitary has shape (2**x_qbits, 2**x_qbits)
    """
    matrix = unitary_group.rvs(2**x_qbits)
    return matrix