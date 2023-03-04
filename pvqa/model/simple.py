from typing import List, Union

from torch import nn

from pvqa.model.base import BaseClassifier


class SimpleClassifier(BaseClassifier):
    def __init__(self, input_dim: int, labels: Union[List[Union[str, int]], int], multilabel: bool = False):
        super().__init__(input_dim, labels, multilabel)
        self.model = nn.Linear(self.input_dim, self.num_classes)
