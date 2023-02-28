import warnings

import pennylane as qml
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from tqdm import TqdmExperimentalWarning

from pvqa.classical_heads import SimpleClassifier
from pvqa.constants import *
from pvqa.data_process import convert_to_post_variational, convert_mnist_data

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

seed_everything(42)

observable_list = list(qml.pauli.pauli_group(4))

labels = ["TShirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "AnkleBoot"]

fmnist_train_dataset = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)
fmnist_test_dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True)
pca_train_dataset, pca_test_dataset = convert_mnist_data(fmnist_train_dataset, fmnist_test_dataset, 16)
pvq_dataset = convert_to_post_variational(pca_train_dataset, 4, observable_list, qml.AmplitudeEmbedding,
                                          embedding_kwargs={"normalize": True}, processing_batch_size=512)
test_dataset = convert_to_post_variational(pca_test_dataset, 4, observable_list, qml.AmplitudeEmbedding,
                                           embedding_kwargs={"normalize": True}, processing_batch_size=512)
val_len = int(len(pvq_dataset) / 10)
train_dataset, val_dataset = random_split(pvq_dataset, [len(pvq_dataset) - val_len, val_len])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

model = SimpleClassifier(len(observable_list), labels, 1e-3)
checkpoint_callback = ModelCheckpoint(monitor="val_f1_score", mode="max", every_n_epochs=1)
tensorboard = pl_loggers.TensorBoardLogger(save_dir=LOG_DIR.joinpath("fashionmnist"))
trainer = Trainer(max_epochs=50, callbacks=[checkpoint_callback], logger=tensorboard)
trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(model, test_dataloader, ckpt_path="best")
