from typing import Iterable, Optional

import pennylane as qml

from pvqa.qencoder.interfaces import QEncoder


class BottomUpEncoder(QEncoder):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], embedding: str,
                 embedding_kwargs: Optional[dict] = None, device: str = "default.qubit", shots: Optional[int] = None):
        super(BottomUpEncoder, self).__init__(n_qubits, observable_list, embedding, embedding_kwargs, device, shots)

        self.model = qml.QNode(self.qnode, self.device, cache=False)

    def qnode(self, features):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        return [qml.expval(observable) for observable in self.observable_list]

    def __call__(self, features):
        return self.model(features).swapaxes(0, 1)
