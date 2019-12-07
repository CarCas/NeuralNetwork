from typing import Sequence
import enum
import numpy as np

from nn.types import Pattern, BaseNeuralNetwork


class ErrorCalculator(enum.Enum):
    MEE = enum.auto()
    MSE = enum.auto()
    MIS = enum.auto()
    ACC = enum.auto()

    def __call__(
        self,
        learning_networks: Sequence[BaseNeuralNetwork],
        patterns: Sequence[Pattern]
    ) -> Sequence[float]:
        if self is ErrorCalculator.MSE:
            error_function = self.mean_square_error
        elif self is ErrorCalculator.MEE:
            error_function = self.mean_euclidean_error
        elif self is ErrorCalculator.MIS:
            error_function = self.mismatch_error
        elif self is ErrorCalculator.ACC:
            error_function = self.accuracy
        else:
            return []

        results = []
        for nn in learning_networks:
            x, d = zip(*patterns)
            results.append(error_function(d, nn(*x)))
        return results

    @staticmethod
    def mean_square_error(
        d: Sequence[float],
        out: Sequence[Sequence[float]]
    ) -> float:
        return np.mean(np.sum(np.square(np.subtract(d, out)), axis=1))

    @staticmethod
    def mean_euclidean_error(
        d: Sequence[float],
        out: Sequence[Sequence[float]]
    ) -> float:
        return np.mean(np.linalg.norm(np.subtract(d, out), axis=1))

    @staticmethod
    def mismatch_error(
        d: Sequence[float],
        out: Sequence[Sequence[float]]
    ) -> float:
        return np.mean(np.not_equal(d, np.round(out)).astype(float))

    @staticmethod
    def accuracy(
        d: Sequence[float],
        out: Sequence[Sequence[float]]
    ) -> float:
        return np.mean(np.equal(d, np.round(out)).astype(float))
