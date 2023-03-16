import warnings
from functools import partial
from typing import Iterable, Optional

from pennylane import numpy as np
import pennylane as qml

from pvqa.qencoder.interfaces import ShadowEncoder

class ShadowBottomUpEncoder(ShadowEncoder):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], embedding: str,
                 embedding_kwargs: Optional[dict] = None, device: str = "default.qubit", shots: Optional[int] = None,
                 strategy: Optional[str] = "qwc", seed: Optional[int] = 42):
        super(ShadowBottomUpEncoder, self).__init__(n_qubits, observable_list, embedding, embedding_kwargs, device,
                                                    shots, strategy, seed)
        self.models = [qml.QNode(partial(self.qnode, observable=observable), self.device, cache=False)
                       for observable in self.random_measurement_basis]

    def qnode(self, features, observable):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        return qml.probs(op=observable)

    def __call__(self, features):
        probs = np.stack([model(features) for model in self.models], axis=0)  # M x B x 2^N
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.sum((self.estimation * probs).transpose([2, 3, 0, 1]) * self.hitmask,
                            axis=(1, 3)) / np.sum(self.hitmask, axis=1)
        result = np.nan_to_num(result, nan=1.0)
        return result