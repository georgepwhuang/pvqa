from typing import Iterable, Optional

import pennylane as qml
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.rich import tqdm

from pvqa.qencoders import BottomUpEncoder, ShadowBottomUpEncoder


def convert_to_post_variational(n_qubits: int,
                                observable_list: Iterable[qml.operation.Observable],
                                embedding: type,
                                train_data: Dataset,
                                test_data: Optional[Dataset],
                                embedding_kwargs: Optional[dict] = None,
                                device: str = "default.qubit",
                                shadow_strategy: Optional[str] = None,
                                shots: Optional[int] = None,
                                processing_batch_size: int = 16):
    if shadow_strategy is None:
        encoder = BottomUpEncoder(n_qubits, observable_list, embedding, embedding_kwargs, device, shots)
    else:
        encoder = ShadowBottomUpEncoder(n_qubits, observable_list, embedding,
                                        embedding_kwargs, device, shots, shadow_strategy)
    train_dataloader = DataLoader(train_data, batch_size=processing_batch_size)
    inputs = []
    labels = []
    for x, y in tqdm(iter(train_dataloader), desc="Converting Training Data to Post Variational Quantum Embeddings"):
        x_mod = encoder(x)
        inputs.append(torch.tensor(x_mod).float())
        labels.append(y.long())
    inputs = torch.cat(inputs)
    labels = torch.cat(labels)
    if test_data is not None:
        test_dataloader = DataLoader(test_data, batch_size=processing_batch_size)
        test_inputs = []
        test_labels = []
        for x, y in tqdm(iter(test_dataloader), desc="Converting Testing Data to Post Variational Quantum Embeddings"):
            x_mod = encoder(x)
            test_inputs.append(torch.tensor(x_mod).float())
            test_labels.append(y.long())
        test_inputs = torch.cat(test_inputs)
        test_labels = torch.cat(test_labels)
        return TensorDataset(inputs, labels), TensorDataset(test_inputs, test_labels), encoder.output_dim
    else:
        return TensorDataset(inputs, labels), encoder.output_dim
