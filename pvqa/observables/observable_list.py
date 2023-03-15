from typing import List
from pennylane.pauli import string_to_pauli_word

from pvqa.observables.base import IterObservables


class ObservableList(IterObservables):
    def __init__(self, qubits: int, observables: List[str]):
        for i in observables:
            assert len(i) == qubits, f'Pauli string must be same length as qubits ({qubits}), the Pauli string has length {len(i)}'
        converted_observables = [string_to_pauli_word(i) for i in observables]
        self.observables = iter(converted_observables)