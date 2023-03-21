import warnings
from functools import partial
from typing import Iterable, Optional

from pennylane import numpy as np
import pennylane as qml
from pvqa.observables.local_observables import LocalPauliObservables

from pvqa.qencoder.interfaces import ShadowEncoder

class ShadowBottomUpEncoder(ShadowEncoder):
    def __init__(self, n_qubits: int, observable_list: Iterable[qml.operation.Observable], embedding: str,
                 embedding_kwargs: Optional[dict] = None, device: str = "default.qubit", shots: Optional[int] = None,
                 strategy: Optional[str] = "qwc", seed: Optional[int] = 42):
        super(ShadowBottomUpEncoder, self).__init__(n_qubits, observable_list, embedding, embedding_kwargs, device,
                                                    shots, strategy, seed)
        self.models = [qml.QNode(partial(self.qnode, observable=observable), self.device, cache=False)
                       for observable in self.random_measurement_basis]
        
        if shots is None:
            self.estimation, self.hitmask = self._generate_estimatation_and_hitmask()
        else:
            self.bitmask = (self.recipe == np.expand_dims(self.observable_tensor, 1)) # L x M x N
            self.hits_bitmask = self.bitmask.sum(axis=-1) > 0 # L x M

    def qnode(self, features, observable):
        self.embedding(features=features, wires=range(self.n_qubits), **self.embedding_kwargs)
        return qml.probs(op=observable)

    def __call__(self, features):
        if self.shots is None:
            probs = np.stack([model(features) for model in self.models], axis=0)  # M x B x 2^N
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = np.sum((self.estimation * probs).transpose([2, 3, 0, 1]) * self.hitmask,
                                axis=(1, 3)) / np.sum(self.hitmask, axis=1)
            result = np.nan_to_num(result, nan=1.0)
            return result
        else:
            one_shot_loc = [model(features).nonzero()[1] for model in self.models]  
            one_shot_loc = np.stack(one_shot_loc, axis=0) # M x B
            binary_mask = 2**np.arange(self.n_qubits-1, -1, -1) # N
            one_shot_loc = np.expand_dims(one_shot_loc, -1) # M x B x 1
            bits = np.bitwise_and(one_shot_loc, binary_mask) != 0 # M x B x N
            eigenval = ((-1)**bits).swapaxes(0,1) # B x M x N
            result = np.expand_dims(eigenval, 1) * self.bitmask # B x L x M x N
            ones = np.ones_like(result)
            result = np.where(result != 0, result, ones) # B x L x M x N
            result = result.prod(axis=-1) # B x L x M
            result = np.sum(result * self.hits_bitmask, axis=-1) / np.maximum(np.sum(self.hits_bitmask, -1), 1)
            return result # B x L
    
if __name__ == "__main__":  
    model = ShadowBottomUpEncoder(4, LocalPauliObservables(4, 2), embedding="AngleEmbedding")
    output = model(np.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]]))
    print(output.shape)