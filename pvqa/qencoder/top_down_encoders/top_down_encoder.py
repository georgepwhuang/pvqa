from functools import partial
from typing import Iterable, Optional, Union

import numpy as np
import pennylane as qml
from pennylane.pauli import group_observables

from pvqa.qencoder.interfaces import TaylorEncoder


class TopDownEncoder(TaylorEncoder):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], derivative_order: int,
                 embedding: str, ansatz: str, embedding_kwargs: Optional[dict] = None, ansatz_kwargs: Optional[dict] = None,
                 device: str = "default.qubit", shots: Optional[int] = None):
        super(TopDownEncoder, self).__init__(n_qubits, observable_list, derivative_order, embedding, ansatz,
                                             embedding_kwargs, ansatz_kwargs, device, shots)
        self.grouped_observables = group_observables(self.observable_list)
        self.models = self._generate_models()

    def _generate_models(self):
        models = {}
        for observable_group in self.grouped_observables:
            models[tuple(observable_group)] = []
            model_set = models[tuple(observable_group)]
            model_set.append(qml.QNode(partial(self.qnode, observables=observable_group), self.device,
                                       diff_method="parameter-shift", max_diff=self.derivative_order))
            for i in range(self.derivative_order):
                model_set.append(qml.jacobian(model_set[i], argnum=1))
        return models

    def qnode(self, features, weights, observables):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        self.ansatz(weights=weights, wires=range(self.n_qubits), **self.ansatz_kwargs)
        return [qml.expval(observable) for observable in observables]

    def __call__(self, features):
        results = []
        for observable_group, models in self.models.items():
            group_results = [model(features, self.init_weight).reshape((len(observable_group), features.shape[0], -1))
                             for model in models]
            group_results = np.concatenate(group_results, axis=-1)
            group_results = group_results.transpose((1, 0, 2))
            results.append(group_results)
        results = np.concatenate(results, axis=1)
        results = results.reshape((features.shape[0], -1))
        return results