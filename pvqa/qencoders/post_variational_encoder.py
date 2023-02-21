from typing import Iterable, Optional

import numpy as np
import pennylane as qml
import torch


class PostVariationalEncoder:
    def __init__(self,
                 n_qubits: int,
                 observable_list: Iterable[qml.operation.Observable],
                 embedding: type,
                 embedding_kwargs: Optional[dict] = None,
                 device: str = "default.qubit",
                 shadow: bool = False,
                 shots: Optional[int] = None):
        self.n_qubits = n_qubits
        self.observable_list = observable_list
        self.device = qml.device(device, wires=self.n_qubits, shots=shots)
        self.shadow = shadow
        self.embedding = embedding
        self.embedding_kwargs = dict() if embedding_kwargs is None else embedding_kwargs

        self.model = qml.QNode(self.qnode, self.device)

    def qnode(self, inputs):
        self.embedding(features=inputs, wires=range(self.n_qubits), **self.embedding_kwargs)
        if self.shadow:
            return qml.shadow_expval(self.observable_list)
        else:
            return [qml.expval(observable) for observable in self.observable_list]

    def __call__(self, *args, **kwargs):
        result = self.model(*args, **kwargs)
        result = torch.Tensor(result)
        return result.transpose(0, 1)


if __name__ == "__main__":
    model = PostVariationalEncoder(2, qml.pauli.pauli_group(2), embedding=qml.AngleEmbedding)
    output = model(np.array([[2, 3], [4, 5], [6, 7]]).astype(np.float64))
