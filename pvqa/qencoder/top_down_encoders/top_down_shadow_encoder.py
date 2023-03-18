import warnings
from functools import partial
from typing import Iterable, Optional

from pennylane import numpy as np
import pennylane as qml

from pvqa.qencoder.interfaces import ShadowEncoder, TaylorEncoder


class ShadowTopDownEncoder(TaylorEncoder, ShadowEncoder):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], derivative_order: int,
                 embedding: str, ansatz: str, embedding_kwargs: Optional[dict] = None,
                 ansatz_kwargs: Optional[dict] = None, device: str = "default.qubit", shots: Optional[int] = None,
                 strategy: str = "qwc", seed: Optional[int] = 42):
        super(ShadowTopDownEncoder, self).__init__(n_qubits=n_qubits, observable_list=observable_list,
                                                   derivative_order=derivative_order, embedding=embedding,
                                                   ansatz=ansatz, embedding_kwargs=embedding_kwargs,
                                                   ansatz_kwargs=ansatz_kwargs, device=device, shots=shots,
                                                   seed=seed, strategy=strategy)
        self.models = self._generate_models()

    def _generate_models(self):
        models = {}
        for idx, observable in enumerate(self.random_measurement_basis):
            models[idx] = []
            model_set = models[idx]
            model_set.append(qml.QNode(partial(self.qnode, observable=observable), self.device,
                                       max_diff=self.derivative_order))
            for i in range(self.derivative_order):
                model_set.append(qml.jacobian(model_set[i], argnum=1))
        return models

    def qnode(self, features, weights, observable):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        self.ansatz(weights=weights, wires=range(self.n_qubits), **self.ansatz_kwargs)
        return qml.probs(op=observable)

    def __call__(self, features):
        probs = []
        for _, models in self.models.items():
            group_results = [model(features, self.init_weight).reshape([features.shape[0], 2 ** self.n_qubits, -1])
                             for model in models]
            group_results = np.concatenate(group_results, axis=-1)  # B x 2^N x D
            group_results = group_results.transpose((2, 0, 1))  # D x B x 2^N
            probs.append(group_results)
        probs = np.expand_dims(np.stack(probs, axis=1), axis=1)  # D x 1 x M x B x 2^N
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prob_est = (self.estimation * probs).transpose([3, 4, 0, 1, 2])  # B x 2^N x D x L x M
            result = np.sum(prob_est * self.hitmask,
                            axis=(1, 4)) / np.sum(self.hitmask, axis=1)
        result = np.nan_to_num(result, nan=1.0)
        result = result.reshape((features.shape[0], -1))
        return result