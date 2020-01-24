from typing import Sequence, Callable, Optional
import numpy as np

from nn.types import BaseNeuralNetwork, Pattern

LearningAlgorithm = Callable[[BaseNeuralNetwork, Sequence[Pattern]], None]


class _LearningAlgorithm:
    def __init__(self, learning_algorithm: LearningAlgorithm, name: Optional[str] = None):
        self.learning_algorithm = learning_algorithm
        self.name = name

    def __call__(self, nn: BaseNeuralNetwork, patterns: Sequence[Pattern]) -> None:
        return self.learning_algorithm(nn, patterns)

    def __repr__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return repr(super())

    def __str__(self) -> str:
        return repr(self)


def _batch(nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
    nn.fit(patterns)


def _online(nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
    for pattern in patterns:
        nn.fit([pattern])


batch = _LearningAlgorithm(_batch, 'batch')
online = _LearningAlgorithm(_online, 'online')


class minibatch:
    def __init__(self, size: float):
        self.size = size

    def __call__(self, nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
        r = round(len(patterns) * self.size)
        r = r if r else 1
        nn.fit(np.random.permutation(patterns)[:r])

    def __repr__(self) -> str:
        return 'minibatch {}'.format(self.size)

    def __str__(self) -> str:
        return repr(self)
