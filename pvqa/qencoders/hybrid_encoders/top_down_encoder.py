from typing import Iterable, Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from pennylane.pauli import group_observables


class TopDownEncoder:
    def __init__(self,
                 n_qubits: int,
                 observable_list: Iterable[qml.operation.Observable],
                 derivative_order: int,
                 embedding: type,
                 ansatz: type,
                 embedding_kwargs: Optional[dict] = None,
                 ansatz_kwargs: Optional[dict] = None,
                 device: str = "default.qubit",
                 shots: Optional[int] = None):
        self.n_qubits = n_qubits
        self.observable_list = list(observable_list)
        self.device = qml.device(device, wires=self.n_qubits, shots=shots)
        self.derivative_order = derivative_order
        self.embedding = embedding
        self.embedding_kwargs = dict() if embedding_kwargs is None else embedding_kwargs
        self.ansatz = ansatz
        self.ansatz_kwargs = dict() if ansatz_kwargs is None else ansatz_kwargs
        
        self.grouped_observables = group_observables(self.observable_list)

        self.model = qml.QNode(self.qnode, self.device, diff_method="parameter-shift", max_diff=self.derivative_order)

    def qnode(self, features, weights, observables):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        self.ansatz(weights=weights, wires=range(self.n_qubits), **self.ansatz_kwargs)
        return [qml.expval(observable) for observable in observables]

    def __call__(self, features, weights):
        results = []
        for observable_group in self.grouped_observables:
            group_results = [self.model(features, weights, observable_group).reshape((len(observable_group), features.shape[0], -1))]
            func = self.model
            for _ in range(self.derivative_order):
                func = qml.jacobian(func, argnum=1)
                group_results.append(func(features, weights, observable_group).reshape((len(observable_group), features.shape[0], -1)))
            group_results = np.concatenate(group_results, axis=-1)
            group_results = group_results.transpose((1, 0, 2))
            results.append(group_results)
        results = np.concatenate(results, axis=1)
        results = results.reshape((features.shape[0], -1))
        return results


if __name__ == "__main__":
    model = TopDownEncoder(2, qml.pauli.pauli_group(2), embedding=qml.AngleEmbedding,
                           ansatz=qml.StronglyEntanglingLayers, derivative_order=2)
    output = model(qnp.tensor([[2, 3], [4, 5], [6, 7]]), qnp.random.rand(1, 2, 3))
