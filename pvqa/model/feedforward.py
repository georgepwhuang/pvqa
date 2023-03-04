from typing import List, Union

from torch import nn

from pvqa.model.base import BaseClassifier


class FeedForwardClassifier(BaseClassifier):
    def __init__(self, input_dim: int, hidden_dim: int, labels: Union[List[Union[str, int]], int], multilabel: bool = False):
        super().__init__(input_dim, labels, multilabel)
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes))
