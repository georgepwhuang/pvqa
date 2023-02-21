import warnings

from matplotlib import pyplot as plt
import pennylane as qml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from sklearn import datasets
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import TqdmExperimentalWarning

from pvqa.constants import *
from pvqa.classical_heads import FeedForwardClassifier
from pvqa.data_process import convert_to_post_variational

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

seed_everything(42)

observable_list = list(qml.pauli.pauli_group(2))

X, y = datasets.make_moons(n_samples=500, shuffle=True, noise=0.2, random_state=42)
inputs = torch.DoubleTensor(X)
labels = torch.DoubleTensor(y)
dataset = TensorDataset(inputs, labels)
pvq_dataset = convert_to_post_variational(dataset, 2, observable_list, qml.IQPEmbedding)
val_len = int(len(pvq_dataset) / 10)
train_dataset, val_dataset = random_split(pvq_dataset, [len(pvq_dataset) - val_len, val_len])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
model = FeedForwardClassifier(len(observable_list), len(observable_list), 2, 1e-3)
checkpoint_callback = ModelCheckpoint(monitor="val_f1_score", mode="max", every_n_epochs=1)
tensorboard = pl_loggers.TensorBoardLogger(save_dir=LOG_DIR.joinpath("moons"))
c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]  # colours for each class
fig = plt.figure()
plt.axis("off")
plt.scatter(X[:, 0], X[:, 1], c=c)
tensorboard.experiment.add_figure(f"dataset", fig)
trainer = Trainer(max_epochs=50, callbacks=[checkpoint_callback], logger=tensorboard)
trainer.fit(model, train_dataloader, val_dataloader)
