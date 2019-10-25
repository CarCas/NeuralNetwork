from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Sequence, Union, Optional

from nn import ActivationFunction


class Neuron:
    def __init__(
        self,
        w: Sequence[float],
        activation: Union[ActivationFunction, sp.Expr, str]
    ):
        if not len(w):
            raise ValueError('w cannot be empty')

        self._w = np.array(w)
        self._activation = ActivationFunction(activation)

        self._state: Optional[Neuron._State] = None

    def __call__(self, *args: float) -> float:
        input = (1,)+tuple(args)
        if len(input) != len(self.w):
            raise ValueError('bad input size')
        net = np.dot(input, self.w)
        out = self.activation(net)
        self._state = Neuron._State(input, net, out)
        return out

    @property
    def w(self) -> Sequence[float]:
        return self._w

    @w.setter
    def w(self, w: Sequence[float]):
        if len(self.w) != len(w):
            raise RuntimeError('w cannot have different size')
        self._w = np.array(w)

    @property
    def activation(self) -> ActivationFunction:
        return self._activation

    @property
    def executed(self) -> bool:
        return self._state is not None

    @property
    def input(self) -> Sequence[float]:
        if self._state is None:
            raise RuntimeError('neuron not executed')
        return self._state.input

    @property
    def net(self) -> float:
        if self._state is None:
            raise RuntimeError('neuron not executed')
        return self._state.net

    @property
    def out(self) -> float:
        if self._state is None:
            raise RuntimeError('neuron not executed')
        return self._state.output

    @property
    def fprime(self) -> float:
        if self._state is None:
            raise RuntimeError('neuron not executed')
        return self.activation.derivative(self.net)

    class _State:
        def __init__(
            self,
            input: Sequence[float],
            net: float,
            output: float
        ):
            self._input = input
            self._net = net
            self._output = output

        @property
        def input(self) -> Sequence:
            return self._input

        @property
        def net(self) -> float:
            return self._net

        @property
        def output(self) -> float:
            return self._output
