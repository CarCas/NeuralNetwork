from typing import Sequence, Callable
from enum import Enum

from nn.types import BaseNeuralNetwork, Pattern

LearningAlgorithm = Callable[[BaseNeuralNetwork, Sequence[Pattern]], None]


def batch(nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
    nn.train(patterns)


def online(nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
    for pattern in patterns:
        nn.train([pattern])
