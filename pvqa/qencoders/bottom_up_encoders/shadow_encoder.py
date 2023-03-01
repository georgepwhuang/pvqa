from typing import Iterable, Optional

import numpy as np
import pennylane as qml
import torch
from functools import partial
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word, group_observables

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
        self.seed = np.random.randint(2 ** 30) if seed is None else seed
        
        self.observable_tensor = self.__generate_observables()
        
        if shots is None:
            self.recipe, self.random_measurement_basis = self.__qwc_generate_shadow_measurements()
        else:
            self.recipe, self.random_measurement_basis = self.__random_generate_shadow_measurements(shots)
            
        self.measurements = len(self.random_measurement_basis)

        self.models = [qml.QNode(partial(self.qnode, observable=observable), self.device, interface="torch", cache=False) 
                       for observable in self.random_measurement_basis]

    def __random_generate_shadow_measurements(self, measurements):
        np.random.seed(self.seed)
        unitary_ids = np.random.randint(0, 3, size=(measurements, self.n_qubits))
        unitaries = np.take(["X", "Y", "Z"], unitary_ids).tolist()
        unitaries = [string_to_pauli_word("".join(p_str)) for p_str in unitaries]
        return torch.tensor(unitary_ids), unitaries

    def __qwc_generate_shadow_measurements(self):
        grouped_observables = group_observables(self.observable_list)
        unitaries = []
        unitary_ids = []
        for observable_group in grouped_observables:
            pauli_char = ['I'] * self.n_qubits
            for observable in observable_group:
                p_str = pauli_word_to_string(observable, wire_map={i: i for i in range(self.n_qubits)})
                pauli_char = [max(pauli_char[i], p_str[i]) for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                if pauli_char[i] == 'I':
                    pauli_char[i] = np.random.choice(["X", "Y", "Z"])
            unitary_ids.append([ord(idx) - ord('X') for idx in pauli_char])
            unitaries.append(string_to_pauli_word("".join(pauli_char)))
        return torch.tensor(unitary_ids), unitaries

    def __generate_observables(self):
        obs_list = []
        for observable in self.observable_list:
            p_str = pauli_word_to_string(observable, wire_map={i: i for i in range(self.n_qubits)})
            p_str = [*p_str]
            obs = torch.tensor(list(map(lambda x: ord(x) - ord('X'), p_str)))
            obs_list.append(obs)
        return torch.stack(obs_list, dim=0)

    def qnode(self, features, observable):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        return qml.probs(op=observable)

    def __call__(self, features):
        torch_device = features.device
        if self.recipe.device != torch_device:
            self.recipe = self.recipe.to(torch_device)
        if self.observable_tensor.device != torch_device:
            self.observable_tensor = self.observable_tensor.to(torch_device)
        one_shot_loc = [model(features).nonzero(as_tuple=True)[1].to(torch_device) for model in self.models] 
        one_shot_loc = torch.stack(one_shot_loc, dim=0) #M x B
        binary_mask = 2**torch.arange(self.n_qubits-1, -1, -1).to(torch_device)
        bits = one_shot_loc.unsqueeze(-1).bitwise_and(binary_mask).ne(0)
        eigenval = (-1)**bits.transpose(0,1)
        bitmask = (self.recipe == self.observable_tensor.unsqueeze(dim=1))
        result = eigenval.unsqueeze(dim = 1) * bitmask
        ones = torch.ones_like(result)
        result = torch.where(result != 0, result, ones)
        hits_bitmask = bitmask.sum(dim=-1).gt(0)
        result = result.prod(dim=-1)
        result = torch.sum(result * hits_bitmask, dim=-1) / torch.clamp(torch.sum(hits_bitmask, -1), min=1)
        return result.cpu()


if __name__ == "__main__":
    model = PyTorchShadowBottomUpEncoder(4, local_pauli_group(4, 2), embedding=qml.AngleEmbedding)
    output = model(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]]).double())