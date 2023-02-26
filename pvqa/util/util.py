from typing import Iterator
from pennylane.pauli import PauliWord, string_to_pauli_word

def local_pauli_group(qubits:int, locality:int) -> Iterator[PauliWord]:
    yield from __generate_paulis(0, 0, "", qubits, locality)
        
def __generate_paulis(identities:int, paulis:int, output:str, qubits:int, locality:int) -> Iterator[PauliWord]:
    if (len(output) == qubits):
        yield string_to_pauli_word(output)

    else:
        yield from __generate_paulis(identities+1, paulis, output+"I", qubits, locality)
        if (paulis < locality):
            yield from __generate_paulis(identities, paulis+1, output+"X", qubits, locality)
            yield from __generate_paulis(identities, paulis+1, output+"Y", qubits, locality)
            yield from __generate_paulis(identities, paulis+1, output+"Z", qubits, locality)
            
if __name__ == "__main__":
    print(list(local_pauli_group(3, 1)))