from abc import ABC, abstractmethod

from matplotlib.figure import Figure
from torch import nn
from typing import Dict, Any, List, Union


class ClassificationMetric(nn.Module, ABC):
    @property
    @abstractmethod
    def scalars(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def nonscalars(self, current_epoch) -> List[Dict[str, Union[str, Figure]]]:
        pass
