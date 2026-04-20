from abc import ABC, abstractmethod
from src.core.candidate import Candidate

class FJSSPAlgorithm(ABC):
    @abstractmethod
    def solve(self, encoding) -> tuple[Candidate, list]:
        pass

    @abstractmethod
    def get_evaluations(self) -> int:
        pass