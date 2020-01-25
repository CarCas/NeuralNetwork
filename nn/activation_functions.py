from typing import Callable, Optional
import numpy as np

from nn.types import ActivationFunction


class SimpleActivationFunction(ActivationFunction):
    def __init__(self, f: Callable, fprime: Callable, name: Optional[str] = None):
        self.f = f
        self.fprime = fprime
        self.name = name

    def __call__(self, x):
        return self.f(x)

    def derivative(self, out):
        return self.fprime(out)

    def __repr__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return repr(super())

    def __str__(self) -> str:
        return repr(self)


class GenericSigmoid(ActivationFunction):
    def __init__(self, a: float):
        self.a = a
        self.name: str = 'sigmoid {}'.format(a)

    def __call__(self, x):
        return 1/(1+np.exp(-self.a*x))

    def derivative(self, out):
        return np.multiply(np.multiply(self.a, out), np.subtract(1, out))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return repr(self)


def _identity(x):
    return x


def _identity_fprime(x):
    return 1


identity = SimpleActivationFunction(_identity, _identity_fprime, 'identity')
sigmoid = GenericSigmoid(a=1)


def _relu(x):
    return (x > 0).astype(int) * x


def _relu_fprime(x):
    return (x > 0).astype(int)


relu = SimpleActivationFunction(_relu, _relu_fprime, 'relu')


def _tanh_fprime(x):
    return 1 - np.square(x)


tanh = SimpleActivationFunction(np.tanh, _tanh_fprime, 'tanh')


def _tanh_classification(x):
    return (np.tanh(x) + 1) / 2


def _tanh_classification_fprime(x):
    return 1 - np.square((x * 2)-1)


tanh_classification = SimpleActivationFunction(_tanh_classification, _tanh_classification_fprime, 'tanh')
