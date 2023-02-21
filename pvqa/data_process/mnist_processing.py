from typing import Optional

from rich.console import Console
import torch
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.datasets import MNIST

from pvqa.constants import DATA_DIR


def convert_mnist_data(train_dataset: MNIST, test_dataset: Optional[MNIST] = None, output_dim: Optional[int] = None):
    data = train_dataset.data
    labels = train_dataset.targets
    if test_dataset is not None:
        test_data = test_dataset.data
        test_labels = test_dataset.targets
        data = torch.concat([data, test_data], dim=0)
    data = data.flatten(1, 2)
    pca = PCA(n_components=("mle" if output_dim is None else output_dim))
    console = Console()
    with console.status("[green]Performing PCA..."):
        data_transform = pca.fit_transform(data)
    console.print("[green]PCA completed")
    data_transform = torch.tensor(data_transform)
    data_transform_normalized = (data_transform - data_transform.mean(dim=0)) / data_transform.std(dim=0)
    if test_dataset is not None:
        test_data_normalized = data_transform_normalized[labels.size(0):]
        data_transform_normalized = data_transform_normalized[:labels.size(0)]
        return TensorDataset(data_transform_normalized, labels), TensorDataset(test_data_normalized, test_labels)
    return TensorDataset(data_transform_normalized, labels)

if __name__ == "__main__":
    train_dataset = datasets.KMNIST(root=DATA_DIR, train=True, download=True)
    output = convert_mnist_data(train_dataset, output_dim=8)
