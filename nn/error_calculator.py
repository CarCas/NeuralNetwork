from typing import Sequence, Callable
import enum
import numpy as np

from nn.types import Pattern, BaseNeuralNetwork
from collections import namedtuple


f_container = namedtuple('f_container', ['f'])


class ErrorCalculator(enum.Enum):
    MSE = f_container(lambda d, out: np.mean(np.sum(np.square(np.subtract(d, out)), axis=1)))
    MEE = f_container(lambda d, out: np.mean(np.linalg.norm(np.subtract(d, out), axis=1)))
    MIS = f_container(lambda d, out: np.mean(np.not_equal(d, np.round(out)).astype(float)))
    ACC = f_container(lambda d, out: np.mean(np.equal(d, np.round(out)).astype(float)))

    def __call__(
        self,
        learning_networks: Sequence[BaseNeuralNetwork],
        patterns: Sequence[Pattern]
    ) -> Sequence[float]:
        error_function = self.value.f
        x, d = zip(*patterns)
        return [error_function(d, nn(*x)) for nn in learning_networks]
