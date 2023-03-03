from typing import List, Union

from torch import nn

from pvqa.classical_heads.base import BaseClassifier


class SimpleClassifier(BaseClassifier):
    def __init__(self, input_dim: int, labels: Union[List[str], int], learning_rate: float):
        super().__init__(input_dim, labels, learning_rate)
        self.model = nn.Linear(self.input_dim, self.num_classes)
