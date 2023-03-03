from typing import Iterable, Optional

import numpy as np
import pennylane as qml

from pvqa.qencoders.interfaces import QEncoder
from pvqa.util import local_pauli_group


class BottomUpEncoder(QEncoder):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], embedding: type,
                 embedding_kwargs: Optional[dict] = None, device: str = "default.qubit", shots: Optional[int] = None):
        super(BottomUpEncoder, self).__init__(n_qubits, observable_list, embedding, embedding_kwargs, device, shots)

        self.model = qml.QNode(self.qnode, self.device, cache=False)

    def qnode(self, features):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        return [qml.expval(observable) for observable in self.observable_list]

    def __call__(self, features):
        return self.model(features).swapaxes(0, 1)


if __name__ == "__main__":
    encoder = BottomUpEncoder(4, local_pauli_group(4, 2), embedding=qml.AngleEmbedding)
    output = encoder(np.array([[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]]))
