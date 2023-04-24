import importlib


class QNN:
    pass
    """
    Interface for CudaQNNs and QuantumQNNs
    """


def get_qnn(qnn_name, x_wires, num_layers, device='cpu'):
    import qnns.cuda_qnn as cuda_qnn
    
    if 'cuda' in qnn_name.lower():
        return getattr(cuda_qnn, qnn_name)(
            num_wires=len(x_wires),
            num_layers=num_layers,
            device=device
        )
    else:
        return getattr(quantum_qnn, qnn_name)(
            wires=x_wires,
            num_layers=num_layers,
            use_torch=True
        )
