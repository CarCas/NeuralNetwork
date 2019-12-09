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


class GenericSigmoid(ActivationFunction):
    def __init__(self, a: float):
        self.a = a

    def __call__(self, x):
        return 1/(1+np.exp(-x))

    def derivative(self, out):
        return self.a * out * (1-out)


identity = SimpleActivationFunction(lambda x: x, lambda _: 1)
sigmoid = GenericSigmoid(a=1)
