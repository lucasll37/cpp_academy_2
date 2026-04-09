from abc import ABC, abstractmethod
from typing import Dict, List


class MiiaModel(ABC):
    """
    Interface mínima para modelos de inferência.

    Inputs e outputs são dicionários de listas de floats.
    """

    @abstractmethod
    def predict(self, inputs: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Executa a inferência.
        """
        pass