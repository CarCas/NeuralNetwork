from typing import Sequence
from enum import Enum
import numpy as np


class ErrorTypes(Enum):
    MEE = 1
    MAE = 2
    MSE = 3
    MIS = 4
    ACC = 5


class ErrorComputation:
    def __init__(self, identifier: ErrorTypes):
        self.identifier: ErrorTypes = identifier

    def __call__(self, d: Sequence[float], out: Sequence[float]) -> Sequence[float]:
        if self.identifier == ErrorTypes.MSE:
            return np.array(self.mean_square_error(d, out))
        elif self.identifier == ErrorTypes.MEE:
            return np.array(self.mean_euclidean_error(d, out))
        elif self.identifier == ErrorTypes.MAE:
            return np.array(self.mean_absolute_error(d, out))
        elif self.identifier == ErrorTypes.MIS:
            return np.array(self.mismatch_error(d, out))
        elif self.identifier == ErrorTypes.ACC:
            return self.accuracy(d, out)
        return np.array(-1)

    def post(self, error: Sequence[float], len: int) -> Sequence[float]:
        if self.identifier == ErrorTypes.MEE:
            error = np.sqrt(error)
        return np.true_divide(error, len)

    @staticmethod
    def mean_square_error(d: Sequence[float], out: Sequence[float]) -> Sequence[float]:
        return np.square(np.subtract(d, out))

    @staticmethod
    def mean_euclidean_error(d: Sequence[float], out: Sequence[float]) -> Sequence[float]:
        return np.square(np.subtract(d, out))

    @staticmethod
    def mean_absolute_error(d: Sequence[float], out: Sequence[float]) -> Sequence[float]:
        return np.abs(np.subtract(d, out))

    @staticmethod
    def mismatch_error(d: Sequence[float], out: Sequence[float]) -> Sequence[float]:
        return np.not_equal(d, np.round(out)).astype(float)

    @staticmethod
    def accuracy(d: Sequence[float], out: Sequence[float]) -> Sequence[float]:
        return np.equal(d, np.round(out)).astype(float)