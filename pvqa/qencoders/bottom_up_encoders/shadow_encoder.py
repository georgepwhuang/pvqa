from typing import Iterable, Optional

import numpy as np
import pennylane as qml
import torch
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word

from pvqa.util import local_pauli_group


class PyTorchShadowBottomUpEncoder:
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

        self.recipe, self.random_measurement_basis = self.__random_generate_shadow_measurements()

        self.observable_tensor = self.__generate_observables()

        self.model = qml.QNode(self.qnode, self.device, interface="torch", cache=False)

    def __random_generate_shadow_measurements(self):
        np.random.seed(self.seed)
        unitary_ids = np.random.randint(0, 3, size=(self.measurements, self.n_qubits))
        unitaries = np.take(["X", "Y", "Z"], unitary_ids).tolist()
        unitaries = [string_to_pauli_word("".join(p_str)) for p_str in unitaries]
        return torch.tensor(unitary_ids), unitaries

    def __qwc_generate_shadow_measurements(self):
        pass

    def __generate_observables(self):
        obs_list = []
        for observable in self.observable_list:
            p_str = pauli_word_to_string(observable, wire_map={i: i for i in range(self.n_qubits)})
            p_str = [*p_str]
            obs = torch.tensor(list(map(lambda x: ord(x) - ord('X'), p_str)))
            obs_list.append(obs)
        return torch.stack(obs_list, dim=0)

    def qnode(self, inputs):
        self.embedding(features=inputs, wires=range(self.n_qubits), **self.embedding_kwargs)
        return qml.counts()

    def __call__(self, inputs, *args, **kwargs):
        torch_device = inputs.device
        if self.recipe.device != torch_device:
            self.recipe = self.recipe.to(torch_device)
        if self.observable_tensor.device != torch_device:
            self.observable_tensor = self.observable_tensor.to(torch_device)
        bits = self.model(inputs).to(torch_device)
        bits = bits.reshape([self.measurements, self.n_qubits, -1])
        bits = bits.permute((2, 0, 1))
        bitmask = (self.recipe == self.observable_tensor.unsqueeze(dim=1))
        result = bits.unsqueeze(dim=1) * bitmask
        ones = torch.ones_like(result)
        result = torch.where(result != 0, result, ones)
        hits_bitmask = bitmask.sum(dim=-1) > 0
        result = result.prod(dim=-1)
        result = torch.sum(result * hits_bitmask, dim=-1) / torch.clamp(torch.sum(hits_bitmask, -1), min=1)
        return result


if __name__ == "__main__":
    model = PyTorchShadowBottomUpEncoder(4, local_pauli_group(4, 2), embedding=qml.AngleEmbedding, shots=100)
    output = model.model(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]]).double())

