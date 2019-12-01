from __future__ import annotations
from typing import Sequence
import numpy as np

from nn.activation_function import ActivationFunction
from nn.number import number


class Neuron:
    def __init__(
        self,
        w: Sequence[number],
        activation: ActivationFunction,
    ):
        if not len(w):
            raise ValueError('w cannot be empty')

        self.w: Sequence[number] = w
        self.activation: ActivationFunction = activation

        self.net: number
        self.out: number
        self.fprime: number

    def __call__(self, *args: number) -> number:
        input_ = (1,) + tuple(args)
        self.net = np.dot(input_, self.w)
        self.out = self.activation(self.net)
        self.fprime = self.activation.derivative(self.net)
        return self.out
