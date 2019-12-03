from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import abc

from nn.neural_network import *
from nn.number import number


class LeariningAlgorthm(abc.ABC):
    def _compute_deltas(self, d: Sequence[number]):
        nn = self.nn

        d = np.array(d)
        output_layer_w = np.array(nn.output_layer.w)
        output_layer_out = np.array(nn)
        hidden_layer_out = np.array((1,) + tuple(nn.hidden_layer.out))[np.newaxis]
        input__layer_out = np.array((1,) + tuple(nn.input_layer))[np.newaxis]

        delta_output = (d - output_layer_out) * nn.output_layer.fprime
        delta_hidden = (output_layer_w.T[1:] @ delta_output) * nn.hidden_layer.fprime

        self.delta_output += (delta_output * hidden_layer_out.T).T
        self.delta_hidden += (delta_hidden * input__layer_out.T).T

    def _update_weights(self):
        nn = self.nn

        nn.output_layer.w += nn.eta * self.delta_output
        nn.hidden_layer.w += nn.eta * self.delta_hidden

        self.delta_output: np.array = 0
        self.delta_hidden: np.array = 0

    def __call__(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        nn: NeuralNetwork
    ):
        self.nn: NeuralNetwork = nn
        self.delta_output: np.array = 0
        self.delta_hidden: np.array = 0

        self._apply(patterns)

    @abc.abstractmethod
    def _apply(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]]
    ):
        pass


class Online(LeariningAlgorthm):
    def _apply(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
    ):
        for x, d in patterns:
            self.nn(*x)
            self._compute_deltas(d)
            self._update_weights()


class Batch(LeariningAlgorthm):
    def _apply(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
    ):
        assert(self.delta_output == 0)
        assert(self.delta_hidden == 0)

        for x, d in patterns:
            self.nn(*x)
            self._compute_deltas(d)

        self.delta_output /= len(patterns)
        self.delta_hidden /= len(patterns)

        self._update_weights()
