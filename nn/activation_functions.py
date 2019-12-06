from typing import Callable
from scipy.special import expit  # type: ignore
import numpy as np

from nn.types import ActivationFunction


class SimpleActivationFunction(ActivationFunction):
    def __init__(self, f: Callable, fprime: Callable):
        self.f = f
        self.fprime = fprime

    def __call__(self, x):
        return self.f(x)

    def derivative(self, out):
        return self.fprime(out)


sign = SimpleActivationFunction(lambda x: float(x > 0), lambda _: 0)
sigmoid = SimpleActivationFunction(expit, lambda out: out * (1 - out))
identity = SimpleActivationFunction(lambda x: x, lambda _: 1)


# sign = ActivationFunction('x>0')
# sigmoid = ActivationFunction('1/(1+exp(-x))')
