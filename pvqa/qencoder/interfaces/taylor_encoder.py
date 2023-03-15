from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import numpy as np
import pennylane as qml
import pvqa.ansatz

from pvqa.qencoder.interfaces.qencoder import QEncoder


class TaylorEncoder(QEncoder, ABC):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], derivative_order: int,
                 embedding: str, ansatz: str, embedding_kwargs: Optional[dict], ansatz_kwargs: Optional[dict],
                 device: str, shots: Optional[int], *args, **kwargs):
        super(TaylorEncoder, self).__init__(n_qubits, observable_list, embedding, embedding_kwargs, device, shots,
                                            *args, **kwargs)
        
        self.derivative_order = derivative_order
        self.ansatz_kwargs = {"layers": 1} if ansatz_kwargs is None else ansatz_kwargs
        try:
            self.ansatz = getattr(qml.templates.layers, ansatz)
            self.init_weight = np.zeros(self.ansatz.shape(self.ansatz_kwargs["layers"], n_qubits))
            self.ansatz_kwargs.pop("layers")
        except AttributeError:
            ansatz_class = getattr(pvqa.ansatz, ansatz)(self.ansatz_kwargs["layers"], n_qubits)
            self.ansatz_kwargs.pop("layers")
            self.ansatz = ansatz_class.ansatz()
            self.init_weight = ansatz_class.weights
        
    @abstractmethod
    def _generate_models(self):
        pass

    @property
    def output_dim(self):
        return len(self.observable_list) * (self.init_weight.size ** (self.derivative_order + 1) - 1) / (
                    self.init_weight.size - 1)