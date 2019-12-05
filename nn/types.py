from typing import Sequence, Tuple, Sequence
import abc

from nn.activation_function import ActivationFunction

Pattern = Tuple[Sequence[float], Sequence[float]]


class NeuralNetwork(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args: float) -> Sequence[float]:
        pass

    @abc.abstractmethod
    def train(self, patterns: Sequence[Pattern]) -> None:
        pass

    @property
    @abc.abstractmethod
    def out(self) -> Sequence[float]:
        pass

    @property
    @abc.abstractmethod
    def input(self) -> Sequence[float]:
        pass


class Architecture(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction,
        eta: float,
    ) -> NeuralNetwork:
        pass
