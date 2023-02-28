import itertools
from typing import Iterable, Optional

import numpy as np
import pennylane as qml
import torch
from pennylane import PauliX, PauliY, PauliZ
from pennylane.pauli import pauli_word_to_string

from pvqa.util import local_pauli_group


class PyTorchShadowPostVariationalEncoder:
    def __init__(self,
                 n_qubits: int,
                 observable_list: Iterable[qml.operation.Observable],
                 embedding: type,
                 embedding_kwargs: Optional[dict] = None,
                 device: str = "default.qubit",
                 shots: Optional[int] = None,
                 seed: Optional[int] = 42):
        self.n_qubits = n_qubits
        self.observable_list = list(observable_list)
        self.device = qml.device(device, wires=self.n_qubits, shots=1)
        self.embedding = embedding
        self.embedding_kwargs = dict() if embedding_kwargs is None else embedding_kwargs
        self.measurements = shots
        self.seed = np.random.randint(2 ** 30) if seed is None else seed

        self.recipe, self.random_measurement_basis = self.__generate_observables()

        self.model = qml.QNode(self.qnode, self.device, interface="torch")

    def __generate_observables(self):
        np.random.seed(self.seed)
        unitary_ids = np.random.randint(0, 3, size=(self.measurements, self.n_qubits))
        unitaries = np.take([PauliX, PauliY, PauliZ], unitary_ids).tolist()
        return torch.tensor(unitary_ids), unitaries

    def qnode(self, inputs):
        self.embedding(features=inputs, wires=range(self.n_qubits), **self.embedding_kwargs)
        return list(itertools.chain(*[[qml.expval(observable(idx)) for idx, observable in enumerate(measurement)]
                                      for measurement in self.random_measurement_basis]))

    def __call__(self, inputs, *args, **kwargs):
        torch_device = inputs.device
        if self.recipe.device != torch_device:
            self.recipe = self.recipe.to(torch_device)
        bits = self.model(inputs, *args, **kwargs)
        bits = bits.reshape([self.measurements, self.n_qubits, -1])
        bits = bits.permute((2, 0, 1))
        results = []
        for observable in self.observable_list:
            p_str = pauli_word_to_string(observable, wire_map={i: i for i in range(self.n_qubits)})
            p_str = [*p_str]
            obs = torch.tensor(list(map(lambda x: ord(x) - ord('X'), p_str)), device=torch_device)
            bitmask = (self.recipe == obs)
            result = bits * bitmask
            result[result == 0] = 1
            hits_bitmask = bitmask.sum(dim=1) > 0
            result = result.prod(dim=2)
            result = torch.sum(result * hits_bitmask, dim=1)/torch.clamp(torch.sum(hits_bitmask), min=1)
            results.append(result)
        return torch.stack(results, dim=1)


if __name__ == "__main__":
    model = PyTorchShadowPostVariationalEncoder(4, local_pauli_group(4, 2), embedding=qml.AngleEmbedding, shots=100)
    output = model(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]]).double())
