from typing import Optional

import pytorch_lightning as pl
import torchvision.datasets
from torch.utils.data import random_split, DataLoader

from pvqa.constants import DATA_DIR
from pvqa.preprocess import convert_mnist_data, encode_quantum_data
from pvqa.qencoder.interfaces import QEncoder

class TorchVisionDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, qdim: int, qencoder: QEncoder, one_hot: bool = False,
                 qnode_batch_size: int = 128, train_batch_size: int = 128):
        super().__init__()
        self.dataset_name = dataset_name
        self.qdim = qdim
        self.qencoder = qencoder
        self.one_hot = one_hot
        self.qnode_batch_size = qnode_batch_size
        self.train_batch_size = train_batch_size
        self.is_setup = False
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.is_setup:
            self.is_setup = True
            try:
                dataset_class = getattr(torchvision.datasets, self.dataset_name.upper())
            except AttributeError:
                raise ModuleNotFoundError(f"Dataset {self.dataset_name.upper()} not supplied in torchvision")
            train_dataset = dataset_class(root=DATA_DIR, train=True, download=True)
            test_dataset = dataset_class(root=DATA_DIR, train=False, download=True)
            pca_train_dataset, pca_test_dataset = convert_mnist_data(train_dataset, test_dataset, self.qdim,
                                                                     one_hot=self.one_hot)
            self.full_dataset, self.test_dataset, self._qencode_dim = \
                encode_quantum_data(self.qencoder,
                                    train_data=pca_train_dataset,
                                    test_data=pca_test_dataset,
                                    processing_batch_size=self.qnode_batch_size)
            val_len = int(len(self.full_dataset) / 10)
            self.train_dataset, self.val_dataset = random_split(self.full_dataset,
                                                                [len(self.full_dataset) - val_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.train_batch_size)

    @property
    def qencode_dim(self):
        return self._qencode_dim
