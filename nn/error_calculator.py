from typing import Sequence, Callable, NamedTuple, Tuple
import enum
import numpy as np

from nn.types import Pattern, BaseNeuralNetwork


ErrorFunction = Callable[[Sequence[BaseNeuralNetwork], Sequence[Sequence[float]]], float]
BestScore = Callable[[Sequence[float]], Tuple[int, float]]


def min_best_score(x: Sequence[float]):
    return np.argmin(x), np.min(x)


def max_best_score(x: Sequence[float]):
    return np.argmax(x), np.max(x)


class ErrorFunctionContainer(NamedTuple):
    f: ErrorFunction
    choose: BestScore


def mse(d, out):
    return np.mean(np.square(np.subtract(d, out)))


def mee(d, out):
    return np.mean(np.linalg.norm(np.subtract(d, out), axis=1))


def mis(d, out):
    return np.mean(np.not_equal(d, np.round(out)).astype(float))


def acc(d, out):
    return np.mean(np.equal(d, np.round(out)).astype(float))


class ErrorCalculator(enum.Enum):
    # mean square error
    MSE = ErrorFunctionContainer(mse, min_best_score)

    # mean euclidean error
    MEE = ErrorFunctionContainer(mee, min_best_score)

    # mismatch
    MIS = ErrorFunctionContainer(mis, min_best_score)

    # accuracy
    ACC = ErrorFunctionContainer(acc, max_best_score)

    def __call__(
        self,
        learning_networks: Sequence[BaseNeuralNetwork],
        patterns: Sequence[Pattern]
    ) -> Sequence[float]:
        error_function: ErrorFunction = self.value.f
        x, d = zip(*patterns)
        return [error_function(d, nn(*x)) for nn in learning_networks]

    @property
    def choose(self) -> BestScore:
        return self.value.choose

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return repr(self)
