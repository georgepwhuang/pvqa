from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.rich import tqdm
from pennylane import numpy as np

from pvqa.qencoder.interfaces import QEncoder


def encode_quantum_data(encoder: QEncoder,
                        train_data: Dataset,
                        test_data: Optional[Dataset],
                        processing_batch_size: int = 16):
    train_dataloader = DataLoader(train_data, batch_size=processing_batch_size)
    inputs = []
    labels = []
    for x, y in tqdm(iter(train_dataloader), desc="Converting Training Data to Post Variational Quantum Embeddings"):
        x_mod = encoder(np.tensor(x))
        inputs.append(torch.tensor(x_mod).float())
        labels.append(y.long())
    inputs = torch.cat(inputs)
    labels = torch.cat(labels)
    if test_data is not None:
        test_dataloader = DataLoader(
            test_data, batch_size=processing_batch_size)
        test_inputs = []
        test_labels = []
        for x, y in tqdm(iter(test_dataloader), desc="Converting Testing Data to Post Variational Quantum Embeddings"):
            x_mod = encoder(np.tensor(x))
            test_inputs.append(torch.tensor(x_mod).float())
            test_labels.append(y.long())
        test_inputs = torch.cat(test_inputs)
        test_labels = torch.cat(test_labels)
        return TensorDataset(inputs, labels), TensorDataset(test_inputs, test_labels), encoder.output_dim
    else:
        return TensorDataset(inputs, labels), encoder.output_dim
