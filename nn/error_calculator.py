from typing import Sequence, Callable, NamedTuple, Tuple
import enum
import numpy as np

from nn.types import Pattern, BaseNeuralNetwork


ErrorFunction = Callable[[Sequence[BaseNeuralNetwork], Sequence[Sequence[float]]], float]
BestScore = Callable[[Sequence[float]], Tuple[int, float]]

min_best_score: BestScore = lambda x: (np.argmin(x), np.min(x))
max_best_score: BestScore = lambda x: (np.argmax(x), np.max(x))


class ErrorFunctionContainer(NamedTuple):
    f: ErrorFunction
    choose: BestScore


class ErrorCalculator(enum.Enum):
    # mean square error
    MSE = ErrorFunctionContainer(lambda d, out: np.mean(np.mean(np.square(np.subtract(d, out)))), min_best_score)

    # mean euclidean error
    MEE = ErrorFunctionContainer(lambda d, out: np.mean(np.linalg.norm(np.subtract(d, out))), min_best_score)

    # mismatch
    MIS = ErrorFunctionContainer(lambda d, out: np.mean(np.not_equal(d, np.round(out)).astype(float)), min_best_score)

    # accuracy
    ACC = ErrorFunctionContainer(lambda d, out: np.mean(np.equal(d, np.round(out)).astype(float)), max_best_score)

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
