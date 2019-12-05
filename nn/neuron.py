from __future__ import annotations
from typing import Sequence
import numpy as np

from nn.activation_function import ActivationFunction


class Neuron:
    def __init__(
        self,
        w: Sequence[float],
        activation: ActivationFunction,
    ):
        if not len(w):
            raise ValueError('w cannot be empty')

        self.w: Sequence[float] = w
        self.activation: ActivationFunction = activation

        self.net: float
        self.out: float
        self.fprime: float

    def __call__(self, *args: float) -> float:
        input_ = (1,) + tuple(args)
        self.net = np.dot(input_, self.w)
        self.out = self.activation(self.net)
        self.fprime = self.activation.derivative(self.net)
        return self.out
