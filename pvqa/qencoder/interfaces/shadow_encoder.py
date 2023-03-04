from abc import ABC
from typing import Iterable, Optional

import numpy as np
import pennylane as qml
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word, group_observables

from pvqa.qencoder.interfaces.qencoder import QEncoder


class ShadowEncoder(QEncoder, ABC):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], embedding: str,
                 embedding_kwargs: Optional[dict], device: str, shots: Optional[int], strategy: str,
                 seed: Optional[int], *args, **kwargs):
        super(ShadowEncoder, self).__init__(n_qubits, observable_list, embedding, embedding_kwargs, device, shots,
                                            *args, **kwargs)
        self.seed = np.random.randint(2 ** 30) if seed is None else seed

        self.observable_tensor = self.__generate_observables()

        if self.shots is None:
            assert strategy == "qwc"
            self.recipe, self.random_measurement_basis = self.__qwc_generate_shadow_measurements()
            self.device = qml.device(device, wires=self.n_qubits)
        else:
            if strategy == "random":
                self.device = qml.device(device, wires=self.n_qubits, shots=1)
                self.recipe, self.random_measurement_basis = self.__random_generate_shadow_measurements()
            elif strategy == "qwc":
                self.recipe, self.random_measurement_basis = self.__qwc_generate_shadow_measurements()
                self.device = qml.device(device, wires=self.n_qubits, shots=shots // len(self.random_measurement_basis))
                self.shots = (shots // len(self.random_measurement_basis)) * len(self.random_measurement_basis)

        self.estimation, self.hitmask = self.__generate_estimatation_and_hitmask()

    def __generate_observables(self):
        obs_list = []
        for observable in self.observable_list:
            p_str = pauli_word_to_string(observable, wire_map={i: i for i in range(self.n_qubits)})
            p_str = [*p_str]
            obs = np.array(list(map(lambda x: ord(x) - ord('X'), p_str)))
            obs_list.append(obs)
        return np.stack(obs_list, axis=0)

    def __random_generate_shadow_measurements(self):
        np.random.seed(self.seed)
        unitary_ids = np.random.randint(0, 3, size=(self.shots, self.n_qubits))
        unitaries = np.take(["X", "Y", "Z"], unitary_ids).tolist()
        unitaries = [string_to_pauli_word("".join(p_str)) for p_str in unitaries]
        return np.array(unitary_ids), unitaries

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
        return np.array(unitary_ids), unitaries

    def __generate_estimatation_and_hitmask(self):
        indexes = np.expand_dims(np.arange(0, 2 ** self.n_qubits), -1)  # 2^N x 1
        binary_mask = 2 ** np.arange(self.n_qubits - 1, -1, -1)  # N
        bits = np.bitwise_and(indexes, binary_mask) != 0  # 2^N x N
        eigenval = (-1) ** bits  # 2^N x N
        bitmask = (self.recipe == np.expand_dims(self.observable_tensor, 1))  # L x M x N
        hitmask = bitmask.sum(-1) > 0  # L x M
        eigen_est = np.expand_dims(bitmask, -2) * eigenval  # L x M x 2^N x N
        ones = np.ones_like(eigen_est)
        est = np.where(eigen_est != 0, eigen_est, ones)  # L x M x 2^N
        est = np.expand_dims(est.prod(axis=-1), axis=-2)  # L x M x 1 x 2^N
        return est, hitmask
