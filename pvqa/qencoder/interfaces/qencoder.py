from abc import ABC, abstractmethod
from typing import Iterable, Optional

import pennylane as qml


class QEncoder(ABC):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], embedding: str,
                 embedding_kwargs: Optional[dict], device: str, shots: Optional[int]):
        super(QEncoder).__init__()
        self.n_qubits = n_qubits
        self.observable_list = list(observable_list)
        self.embedding = getattr(qml.templates.embeddings, embedding)
        self.embedding_kwargs = dict() if embedding_kwargs is None else embedding_kwargs
        self.device = qml.device(device, wires=self.n_qubits, shots=shots)
        self.shots = shots

    @abstractmethod
    def qnode(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @property
    def output_dim(self):
        return len(self.observable_list)
