import random
from typing import Sequence, Callable

from nn.types import BaseNeuralNetwork, Pattern

LearningAlgorithm = Callable[[BaseNeuralNetwork, Sequence[Pattern]], None]


def batch(nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
    nn.train(patterns)


def online(nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
    for pattern in patterns:
        nn.train([pattern])


class minibatch:
    def __init__(self, size: float):
        self.size = size

    def __call__(self, nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
        r = round(len(patterns) * self.size)
        r = r if r else 1
        nn.train(random.sample(patterns, r))
