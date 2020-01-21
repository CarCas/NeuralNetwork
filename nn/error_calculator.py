from typing import Sequence, Callable, NamedTuple
import enum
import numpy as np

from nn.types import Pattern, BaseNeuralNetwork


ErrorFunction = Callable[[Sequence[BaseNeuralNetwork], Sequence[Sequence[float]]], float]


class ErrorFunctionContainer(NamedTuple):
    f: ErrorFunction


class ErrorCalculator(enum.Enum):
    # mean square error
    MSE = ErrorFunctionContainer(lambda d, out: np.mean(np.mean(np.square(np.subtract(d, out)))))

    # mean euclidean error
    MEE = ErrorFunctionContainer(lambda d, out: np.mean(np.linalg.norm(np.subtract(d, out))))

    # mismatch
    MIS = ErrorFunctionContainer(lambda d, out: np.mean(np.not_equal(d, np.round(out)).astype(float)))

    # accuracy
    ACC = ErrorFunctionContainer(lambda d, out: np.mean(np.equal(d, np.round(out)).astype(float)))

    def __call__(
        self,
        learning_networks: Sequence[BaseNeuralNetwork],
        patterns: Sequence[Pattern]
    ) -> Sequence[float]:
        error_function: ErrorFunction = self.value.f
        x, d = zip(*patterns)
        return [error_function(d, nn(*x)) for nn in learning_networks]
