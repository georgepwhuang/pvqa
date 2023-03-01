from typing import Iterable, Optional

import pennylane as qml
import torch


class PyTorchBottomUpEncoder:
    def __init__(self,
                 n_qubits: int,
                 observable_list: Iterable[qml.operation.Observable],
                 embedding: type,
                 embedding_kwargs: Optional[dict] = None,
                 device: str = "default.qubit",
                 shots: Optional[int] = None):
        self.n_qubits = n_qubits
        self.observable_list = list(observable_list)
        self.device = qml.device(device, wires=self.n_qubits, shots=shots)
        self.embedding = embedding
        self.embedding_kwargs = dict() if embedding_kwargs is None else embedding_kwargs

        self.model = qml.QNode(self.qnode, self.device, interface="torch", cache=False)

    def qnode(self, inputs):
        self.embedding(features=inputs, wires=range(self.n_qubits), **self.embedding_kwargs)
        return [qml.expval(observable) for observable in self.observable_list]

    def __call__(self, inputs):
        return self.model(inputs).reshape([len(self.observable_list), -1]).transpose(0, 1)


if __name__ == "__main__":
    model = PyTorchBottomUpEncoder(2, qml.pauli.pauli_group(2), embedding=qml.AngleEmbedding)
    output = model(torch.tensor([[2, 3], [4, 5], [6, 7]]).double())
