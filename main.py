import pennylane as qml
import numpy as np
from pennylane.transforms import *

from pvqa.ansatz.strongly_entangled_layer import IdentityOriginStronglyEntanglingLayer
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def qfunc(parameters):
    IdentityOriginStronglyEntanglingLayer(weights=parameters, wires=range(4))
    return qml.expval(qml.PauliZ(0))

weights = np.zeros([4,4,3])
print(qml.draw(qfunc, expansion_strategy="device")(weights))