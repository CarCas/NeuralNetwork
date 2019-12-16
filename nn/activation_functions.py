from typing import Callable
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
        return 1/(1+np.exp(-self.a*x))

    def derivative(self, out):
        return np.multiply(np.multiply(self.a, out), np.subtract(1, out))


identity = SimpleActivationFunction(lambda x: x, lambda _: 1)
sigmoid = GenericSigmoid(a=1)
