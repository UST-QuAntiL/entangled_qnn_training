import torch
import numpy as np

# Gate implementations of common quantum gates for simulation

small_I = None
other_dig_j = None
other_dig_one_and_minus_one = None
one_top_left = None
one_top_left = None
one_bottom_right = None
_H = None
_CNOT = None


def init_globals(device='cpu'):
    global small_I
    global other_dig_j
    global other_dig_one_and_minus_one
    global one_top_left
    global one_top_left
    global one_bottom_right
    global _H
    global _CNOT
    small_I = torch.tensor([
        [1, 0],
        [0, 1]
    ], dtype=torch.complex128, device=device)

    other_dig_j = torch.tensor([
        [0, 1j],
        [1j, 0]
    ], dtype=torch.complex128, device=device)

    other_dig_one_and_minus_one = torch.tensor([
        [0, -1],
        [1, 0]
    ], dtype=torch.complex128, device=device)

    one_top_left = torch.tensor([
        [1, 0],
        [0, 0]
    ], dtype=torch.complex128, device=device)

    one_bottom_right = torch.tensor([
        [0, 0],
        [0, 1]
    ], dtype=torch.complex128, device=device)

    _H = torch.tensor([
        [1. / np.sqrt(2.), 1. / np.sqrt(2.)],
        [1. / np.sqrt(2.), -1. / np.sqrt(2.)]
    ], dtype=torch.complex128, device=device)

    _CNOT = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.complex128, device=device)


init_globals(device='cpu')


def I(size=2, device='cpu'):
    return torch.eye(size, dtype=torch.complex128, device=device)


def CNOT():
    return _CNOT


def RX(rx):
    x_sin = torch.sin(rx / 2.)
    x_cos = torch.cos(rx / 2.)
    result = small_I*x_cos - other_dig_j*x_sin
    return result


def RY(ry):
    y_sin = torch.sin(ry / 2.)
    y_cos = torch.cos(ry / 2.)
    result = small_I*y_cos + other_dig_one_and_minus_one*y_sin
    return result


def RZ(rz):
    z_exp = torch.exp(1j*rz)
    return one_top_left + z_exp*one_bottom_right


def U3(rx, ry, rz):
    x_cos = torch.cos(rx / 2.) + 0j
    x_sin = torch.sin(rx / 2.) + 0j
    y_exp = torch.exp(1j * ry) + 0j
    z_exp = torch.exp(1j * rz) + 0j
    return torch.stack([x_cos, -z_exp * x_sin, y_exp * x_sin, z_exp * y_exp * x_cos]).reshape((2, 2))
    # return torch.matmul(torch.matmul(RZ(rz), RY(ry)), RX(rx))


def controlled_U(c_wire, t_wire, U):
    if c_wire < t_wire:
        diff = t_wire - c_wire
        cu_size = 2 << diff  # 2**involved_qubits = 2**(diff+1) = 4**(diff) = 2 << diff
        CU = torch.eye(cu_size, dtype=torch.complex128)
        u_size = U.shape[0]
        half_cu_size = cu_size >> 1
        for i in range(half_cu_size, CU.shape[0], u_size):
            CU[i:i+u_size, i:i+u_size] = U
    elif c_wire > t_wire:
        I_size = 1 << (c_wire - t_wire)
        UoI = torch.kron(U, I(I_size))
        cu_size = I_size << 1
        CU = I(cu_size)
        for i in range(1, len(CU), 2):
            CU[i] = UoI[i]
    else:
        raise ValueError("control and target qubit cannot be the same for controlled operations!")
    return CU


def H():
    return _H


def is_unitary(M, error=1e-15):
    zeros = (M @ M.T.conj()) - I(size=M.shape[0])
    return ((zeros.real < error).all and (zeros.imag < error).all()).item()


def quick_all_matmul_vec(M, X, device='cpu'):
    # global matmulvec_result
    matmulvec_result = torch.empty(X.shape, dtype=torch.complex128, device=device)
    size = M.shape[0]
    for idx in range(len(X)):
        for i in range(0, matmulvec_result.shape[1], size):
            matmulvec_result[idx, i:i+size] = torch.matmul(M, X[idx, i:i+size])
    return matmulvec_result


def quick_matmulvec(M, vec, device='cpu'):
    matmulvec_result = torch.empty(vec.shape, dtype=torch.complex128, device=device)
    size = M.shape[0] #quadratic, no care

    for i in range(0, vec.shape[0], size):
        matmulvec_result[i:i+size] = torch.matmul(M, vec[i:i+size])
    return matmulvec_result


def quick_matmulmat(A, B, device='cpu'):
    """
    To do: A*(IxB) = X*(IxU.T)
    """
    matmulmat_result = torch.empty(A.shape, dtype=torch.complex128, device=device)
    size = B.shape[0]   # 2**x_qbit in case B=U.T
    for i in range(0, A.shape[0], size):
        for j in range(0, A.shape[1], size):
            matmulmat_result[i:i+size, j:j+size] = torch.matmul(A[i:i+size, j:j+size], B)
    return matmulmat_result


def main():
    U = torch.tensor([
        [2, 2],
        [2, 2]
    ])
    cu = controlled_U(2, 0, U)
    print(cu)


if __name__ == '__main__':
    main()
