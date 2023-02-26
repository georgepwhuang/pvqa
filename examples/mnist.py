import os
os.environ["OMP_NUM_THREADS"] = "4"
import warnings

import pennylane as qml
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from tqdm import TqdmExperimentalWarning

from pvqa.classical_heads.simple import SimpleClassifier
from pvqa.constants import *
from pvqa.data_process import convert_to_post_variational, convert_mnist_data
from pvqa.util import local_pauli_group

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

seed_everything(42)

n_qubits = 5
q_dim = 2**5
qnode_batch_size = 128
train_batch_size = 128

observable_list = list(local_pauli_group(n_qubits, 3))

labels = list(map(lambda x: str(x), range(10)))

kmnist_train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True)
kmnist_test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True)
pca_train_dataset, pca_test_dataset = convert_mnist_data(kmnist_train_dataset, kmnist_test_dataset, q_dim)
pvq_dataset = convert_to_post_variational(pca_train_dataset, n_qubits, observable_list, qml.AmplitudeEmbedding,
                                          embedding_kwargs={"normalize": True}, processing_batch_size=qnode_batch_size, 
                                          shadow=True, shots=1000)
test_dataset = convert_to_post_variational(pca_test_dataset, n_qubits, observable_list, qml.AmplitudeEmbedding,
                                           embedding_kwargs={"normalize": True}, processing_batch_size=qnode_batch_size, 
                                           shadow=True, shots=1000)
val_len = int(len(pvq_dataset) / 10)
train_dataset, val_dataset = random_split(pvq_dataset, [len(pvq_dataset) - val_len, val_len])

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=train_batch_size)

model = SimpleClassifier(len(observable_list), labels, 1e-3)
checkpoint_callback = ModelCheckpoint(monitor="val_f1_score", mode="max", every_n_epochs=1)
tensorboard = pl_loggers.TensorBoardLogger(save_dir=LOG_DIR.joinpath("mnist"))
trainer = Trainer(max_epochs=50, callbacks=[checkpoint_callback], logger=tensorboard)
trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(model, test_dataloader, ckpt_path="best")
