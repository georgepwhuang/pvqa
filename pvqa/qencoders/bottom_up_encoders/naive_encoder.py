from typing import Iterable, Optional

import pennylane as qml
from pennylane import numpy as qnp


class BottomUpEncoder:
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

        self.model = qml.QNode(self.qnode, self.device, cache=False)

    def qnode(self, features):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        return [qml.expval(observable) for observable in self.observable_list]

    def __call__(self, features):
        return self.model(features)


if __name__ == "__main__":
    model = BottomUpEncoder(2, qml.pauli.pauli_group(2), embedding=qml.AngleEmbedding)
    output = model(qnp.tensor([[2,3],[4,5]]))
    print(output)