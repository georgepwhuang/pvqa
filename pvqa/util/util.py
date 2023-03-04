from typing import Iterator, Type
from pennylane.operation import Observable
from pennylane.pauli import string_to_pauli_word, pauli_group

from pvqa import qencoder
from pvqa.qencoder.interfaces import QEncoder


class LocalPauliObservables:
    def __init__(self, qubits: int, locality: int):
        self.observables = local_pauli_group(qubits, locality)

    def __iter__(self) -> Iterator[Observable]:
        return self.observables


def local_pauli_group(qubits: int, locality: int) -> Iterator[Observable]:
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

if __name__ == "__main__":
    print(list(local_pauli_group(3, 1)))
