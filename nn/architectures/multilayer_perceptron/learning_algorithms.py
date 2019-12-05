from __future__ import annotations
from typing import Sequence
import numpy as np
import abc

from nn.types import Pattern
from nn.architectures.multilayer_perceptron.types import MLPBaseNeuralNetwork


class LeariningAlgorthm(abc.ABC):
    def _compute_deltas(self, d: Sequence[float]):
        nn: MLPBaseNeuralNetwork = self.nn

        # SetUp
        d = np.array(d)
        output_layer_w = np.array(nn.output_layer.w)
        hidden_layers_w = [layer.w for layer in nn.hidden_layers]

        output_layer_out = np.array(nn.output_layer.out)
        hidden_layers_out = [
            np.array((1,) + tuple(layer.out))[np.newaxis]
            for layer in nn.hidden_layers
        ]
        input__layer_out = np.array((1,) + tuple(nn.input))[np.newaxis]

        deltas_hidden = [0] * len(nn.hidden_layers)

        # Gradient calculation
        delta_output = (d - output_layer_out) * nn.output_layer.fprime
        deltas_hidden[-1] = (output_layer_w.T[1:] @ delta_output) * nn.hidden_layers[-1].fprime
        for i in range(0, len(hidden_layers_w)-1)[::-1]:
            deltas_hidden[i] = ((np.array(hidden_layers_w[i+1]).T[1:] @ deltas_hidden[i+1]) * nn.hidden_layers[i].fprime)

        self.gradients_hidden[0] += (deltas_hidden[0] * input__layer_out.T).T
        for i in range(1, len(hidden_layers_out)):
            self.gradients_hidden[i] += (deltas_hidden[i] * hidden_layers_out[i-1].T).T
        self.gradient_output += (delta_output * hidden_layers_out[-1].T).T

    def _update_weights(self):
        nn = self.nn

        nn.output_layer.w += self.eta * self.gradient_output
        self.gradient_output = 0

        for i in range(len(nn.hidden_layers)):
            nn.hidden_layers[i].w += self.eta * self.gradients_hidden[i]
            self.gradients_hidden[i] = 0

    def __call__(
        self,
        patterns: Sequence[Pattern],
        nn: MLPBaseNeuralNetwork,
        eta: float,
    ):
        self.eta: float = eta
        self.nn: MLPBaseNeuralNetwork = nn
        self.gradient_output: np.array = 0
        self.gradients_hidden: np.array = [0] * len(nn.hidden_layers)

        self._apply(patterns)

    @abc.abstractmethod
    def _apply(
        self,
        patterns: Sequence[Pattern]
    ):
        pass


class Online(LeariningAlgorthm):
    def _apply(
        self,
        patterns: Sequence[Pattern],
    ):
        for x, d in patterns:
            self.nn(*x)
            self._compute_deltas(d)
            self._update_weights()


class Batch(LeariningAlgorthm):
    def _apply(
        self,
        patterns: Sequence[Pattern],
    ):
        for x, d in patterns:
            self.nn(*x)
            self._compute_deltas(d)

        self.gradient_output /= len(patterns)
        for i in range(len(self.gradients_hidden)):
            self.gradients_hidden[i] /= len(patterns)

        self._update_weights()
