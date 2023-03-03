from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import numpy as np
import pennylane as qml

from pvqa.qencoders.interfaces.qencoder import QEncoder


class TaylorEncoder(QEncoder, ABC):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], derivative_order: int,
                 embedding: type, ansatz: type, init_weight: Union[np.array, qml.numpy.tensor],
                 embedding_kwargs: Optional[dict], ansatz_kwargs: Optional[dict],
                 device: str, shots: Optional[int], *args, **kwargs):
        super(TaylorEncoder, self).__init__(n_qubits, observable_list, embedding, embedding_kwargs, device, shots,
                                            *args, **kwargs)
        self.init_weight = init_weight
        self.derivative_order = derivative_order
        self.ansatz = ansatz
        self.ansatz_kwargs = dict() if ansatz_kwargs is None else ansatz_kwargs

    @abstractmethod
    def _generate_models(self):
        pass

    @property
    def output_dim(self):
        return len(self.observable_list) * (self.init_weight.size ** (self.derivative_order + 1) - 1) / (
                    self.init_weight.size - 1)