from __future__ import annotations
import numpy as np
from typing import Sequence, Optional, Callable, Tuple

from nn import ActivationFunction
from nn.neuron_layer import NeuronLayer


def back_propagation(
    d: Sequence[float],
    eta: float,
    nn: NeuralNetwork,
):
    delta_k = []
    for k in range(len(nn.output_layer)):
        n_k = nn.output_layer[k]
        delta_k.append((d[k] - n_k.out) * n_k.fprime)

        delta_w = [0]
        for i in range(len(nn.hidden_layer)):
            o_i = nn.hidden_layer[i].out
            delta_w.append(delta_k[-1] * o_i)
        n_k.w += np.multiply(eta, delta_w)

    delta_j = []
    for j in range(len(nn.hidden_layer)):
        n_j = nn.hidden_layer[j]
        wk_j = nn.output_layer.w_from(j)
        delta_j.append(np.dot(delta_k, wk_j) * n_j.fprime)

        delta_w = [0]
        for i in range(len(nn.input_layer)):
            o_i = nn.input_layer[i]
            delta_w.append(delta_j[-1] * o_i)
        n_j.w += np.multiply(eta, delta_w)


class NeuralNetwork:
    def __init__(
        self,
        activation: ActivationFunction,
        architecture: Architecture,
        learning_algorithm: Callable = back_propagation
    ):
        self.activation = activation
        self.learning_algorithm = learning_algorithm

        self._layers = [
            NeuronLayer(
                size=architecture.number_hidden,
                activation=activation,
                weights=architecture.hidden_weights),
            NeuronLayer(
                size=architecture.number_outputs,
                activation=activation,
                weights=architecture.output_weights),
        ]

    def __call__(self, *args: float) -> Sequence[float]:
        self._input = tuple(args)

        output = self._input
        for layer in self._layers:
            output = layer(*output)
        return output

    @property
    def out(self) -> Sequence[float]:
        return self.output_layer.out

    @property
    def input_layer(self) -> Sequence[float]:
        return self._input

    @property
    def output_layer(self) -> NeuronLayer:
        return self._layers[-1]

    @property
    def hidden_layer(self) -> NeuronLayer:
        return self._layers[0]

    def train(
        self,
        patterns: Sequence[Tuple[Sequence[float], Sequence[float]]],
        epoch_number: int,
        eta: float = 0.1
    ):
        for _ in range(epoch_number):
            for x, d in patterns:
                self(*x)
                self.learning_algorithm(d, eta, self)

    def test(
        self,
        patterns: Sequence[Tuple[Sequence[float], Sequence[float]]],

    ) -> float:
        error = 0
        for x, d in patterns:
            error += (np.subtract(self(*x), d)**2).sum()/2

        return error

    class Architecture:
        def __init__(
            self,
            number_inputs: int,
            number_hidden: int,
            number_outputs: int,

            hidden_weights: Optional[Sequence[Sequence[float]]] = None,
            output_weights: Optional[Sequence[Sequence[float]]] = None,
        ):
            if number_inputs < 1 or number_hidden < 1 or number_outputs < 1:
                raise ValueError('number_* cattot be lesser than 1')

            self._number_inputs = number_inputs
            self._number_hidden = number_hidden
            self._number_outputs = number_outputs

            if hidden_weights is None:
                self._hidden_weights = (((0.1,) * (self.number_inputs + 1),)
                                        * self.number_hidden)
            else:
                if len(hidden_weights) != number_hidden:
                    raise ValueError('len(hidden_weights) != number_hidden')
                for i, w in enumerate(hidden_weights):
                    if len(w) != number_inputs + 1:
                        raise ValueError('len(hidden_weights['
                                         + str(i) + ']) != number_inputs + 1')
                self._hidden_weights = hidden_weights

            if output_weights is None:
                self._output_weights = (((0.1,) * (self.number_hidden + 1),)
                                        * self.number_outputs)
            else:
                if len(output_weights) != number_outputs:
                    raise ValueError('len(output_weights) != number_outputs')
                for i, w in enumerate(output_weights):
                    if len(w) != number_hidden + 1:
                        raise ValueError('len(output_weights['
                                         + str(i) + ']) != number_hidden + 1')
                self._output_weights = output_weights

        @property
        def number_inputs(self):
            return self._number_inputs

        @property
        def number_hidden(self):
            return self._number_hidden

        @property
        def number_outputs(self):
            return self._number_outputs

        @property
        def hidden_weights(self):
            return self._hidden_weights

        @property
        def output_weights(self):
            return self._output_weights
