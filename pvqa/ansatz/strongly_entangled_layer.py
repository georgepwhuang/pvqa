from pennylane import numpy as np
import pennylane as qml

from pvqa.ansatz.base import Ansatz

class IdentityOriginStronglyEntanglingLayers(Ansatz):
    def __init__(self, layers, n_qubits):
        self.layers = layers
        self._weights = np.zeros([2 * layers, n_qubits, 3])
        
    def ansatz(self):
        def ansatz_func(weights, wires, ranges=None, imprimitive=None):
            shape = qml.math.shape(weights)[-3:]
            wires = list(wires)
            assert shape[0] % 2 == 0, f"Weights tensor must have first dimension divisible by 2; got {shape[0]}"
            if ranges is None:
                if len(wires) > 1:
                    ranges = [(l % (len(wires) - 1)) + 1 for l in range(self.layers)]
                else:
                    ranges = [0] * self.layers
            reverse_ranges = list(reversed([len(wires) - i for i in ranges]))
            qml.StronglyEntanglingLayers(weights[:self.layers], wires, ranges, imprimitive)
            qml.StronglyEntanglingLayers(weights[self.layers:], list(reversed(wires)), reverse_ranges, imprimitive)
        return ansatz_func

    @property
    def weights(self):
        return self._weights
