from abc import ABC, abstractmethod
class Ansatz(ABC):
    @property
    @abstractmethod
    def weights(self):
        pass
    
    @abstractmethod
    def ansatz(self):
        pass