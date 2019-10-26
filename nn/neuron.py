from __future__ import annotations
from typing import List
import math

from nn.activation_function import ActivationFunction


class Neuron:
    def __init__(
        self,
        activation: ActivationFunction,
        w: List[float],
        bias: float
    ):
        if not len(w):
            raise ValueError('w cannot be empty')

        self.bias: float = bias
        self.w: List[float] = w
        self.activation: ActivationFunction = activation

        self.net: float
        self.out: float

    def __call__(self, *args: float) -> float:
        input = tuple(args)
        self.net = 0
        for i in range(len(input)):
            self.net += input[i] * self.w[i]
        self.net += self.bias
        self.out = self.activation(self.net)
        return self.out

    @property
    def fprime(self) -> float:
        return self.activation.derivative(self.net)
