from typing import Iterator, Optional
from pennylane.operation import Observable
from pennylane.pauli import string_to_pauli_word, pauli_group

from pvqa.observables.base import IterObservables


class LocalPauliObservables(IterObservables):
    def __init__(self, qubits: int, locality: Optional[int]=None):
        if locality is None:
            locality = qubits
        self.observables = local_pauli_group(qubits, locality)


def local_pauli_group(qubits: int, locality: int) -> Iterator[Observable]:
    assert locality <= qubits, f'Locality must not exceed the number of qubits.'
    if locality == qubits:
        yield from pauli_group(qubits)
    else:
        yield from __generate_paulis(0, 0, "", qubits, locality)


def __generate_paulis(identities: int, paulis: int, output: str, qubits: int, locality: int) -> Iterator[Observable]:
    if len(output) == qubits:
        yield string_to_pauli_word(output)

    else:
        yield from __generate_paulis(identities + 1, paulis, output + "I", qubits, locality)
        if paulis < locality:
            yield from __generate_paulis(identities, paulis + 1, output + "X", qubits, locality)
            yield from __generate_paulis(identities, paulis + 1, output + "Y", qubits, locality)
            yield from __generate_paulis(identities, paulis + 1, output + "Z", qubits, locality)