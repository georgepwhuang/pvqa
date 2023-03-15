from abc import ABC
from pennylane.operation import Observable
from typing import Iterator

class IterObservables(ABC):
    def __iter__(self) -> Iterator[Observable]:
        return self.observables