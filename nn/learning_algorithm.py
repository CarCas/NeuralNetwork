from typing import Sequence
from enum import Enum, auto

from nn.types import BaseNeuralNetwork, Pattern


class LearningAlgorithm(Enum):
    ONLINE = auto()
    BATCH = auto()

    def __call__(self, nn: BaseNeuralNetwork, patterns: Sequence[Pattern]):
        if self is self.BATCH:
            nn.train(patterns)

        elif self is self.ONLINE:
            for pattern in patterns:
                nn.train([pattern])
