from typing import Iterable, Optional

import pennylane as qml
from pennylane import numpy as np
from pennylane.pauli import group_observables
from functools import partial

from pvqa.qencoder.interfaces import QEncoder


class BottomUpEncoder(QEncoder):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], embedding: str,
                 embedding_kwargs: Optional[dict] = None, device: str = "default.qubit", shots: Optional[int] = None):
        super(BottomUpEncoder, self).__init__(n_qubits, observable_list, embedding, embedding_kwargs, device, shots)

        self.grouped_observables = group_observables(self.observable_list)
        self.models = self._generate_models()

    def _generate_models(self):
        models = {}
        for observable_group in self.grouped_observables:
            models[tuple(observable_group)] = []
            model_set = models[tuple(observable_group)]
            model_set.append(qml.QNode(partial(self.qnode, observables=observable_group), self.device))
        return models

    def qnode(self, features, observables):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        return [qml.expval(observable) for observable in observables]

    def __call__(self, features):
        results = []
        for observable_group, models in self.models.items():
            group_results = [model(features).reshape((len(observable_group), features.shape[0], -1))
                             for model in models]
            group_results = np.concatenate(group_results, axis=-1)
            group_results = group_results.transpose((1, 0, 2))
            results.append(group_results)
        results = np.concatenate(results, axis=1)
        results = results.reshape((features.shape[0], -1))
        return results
