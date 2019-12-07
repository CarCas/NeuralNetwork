from typing import Sequence, Tuple, Sequence, Any
import abc

Pattern = Tuple[Sequence[float], Sequence[float]]


class ActivationFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Any) -> Any:
        pass

    @abc.abstractmethod
    def derivative(self, out: Any) -> Any:
        pass


class BaseNeuralNetwork(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args: Sequence[float]) -> Sequence[Sequence[float]]:
        pass

    @abc.abstractmethod
    def train(self, patterns: Sequence[Pattern]) -> None:
        pass


class Architecture(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction,
        eta: float,
    ) -> BaseNeuralNetwork:
        pass
