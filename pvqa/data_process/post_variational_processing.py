from typing import Iterable, Optional

import pennylane as qml
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.rich import tqdm

from pvqa.qencoders import PyTorchPostVariationalEncoder, PyTorchShadowPostVariationalEncoder


def convert_to_post_variational(data: Dataset,
                                n_qubits: int,
                                observable_list: Iterable[qml.operation.Observable],
                                embedding: type,
                                embedding_kwargs: Optional[dict] = None,
                                device: str = "default.qubit",
                                shadow: bool = False,
                                shots: Optional[int] = None,
                                processing_batch_size: int = 16):
    if shadow:
        encoder = PyTorchShadowPostVariationalEncoder(n_qubits, observable_list, embedding, embedding_kwargs, device,
                                                      shots)
    else:
        encoder = PyTorchPostVariationalEncoder(n_qubits, observable_list, embedding, embedding_kwargs,
                                                device, shots)
    dataloader = DataLoader(data, batch_size=processing_batch_size)
    inputs = []
    labels = []
    for x, y in tqdm(iter(dataloader), desc="Converting to Post Variational Quantum Embeddings"):
        x_mod = encoder(x)
        inputs.append(x_mod)
        labels.append(y.long())
    inputs = torch.cat(inputs)
    labels = torch.cat(labels)
    return TensorDataset(inputs, labels)
