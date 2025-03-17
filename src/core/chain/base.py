from abc import ABC, abstractmethod

class BaseChainUnit(ABC):
    @abstractmethod
    def invoke(self, query: str):
        pass
