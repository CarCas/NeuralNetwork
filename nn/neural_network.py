from __future__ import annotations
from typing import Sequence, Optional, Callable, Tuple, List
import random

from nn.activation_function import ActivationFunction
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

    delta_j = []
    for j in range(len(nn.hidden_layer)):
        n_j = nn.hidden_layer[j]
        wk_j = nn.output_layer.w_from(j)
        delta = 0
        for i in range(len(delta_k)):
            delta += delta_k[i] * wk_j[i]
        delta_j.append(delta * n_j.fprime)

    for k in range(len(nn.output_layer)):
        n_k = nn.output_layer[k]
        delta_w = []
        for i in range(len(nn.hidden_layer)):
            o_i = nn.hidden_layer[i].out
            delta_w.append(delta_k[k] * o_i)
        for i in range(len(delta_w)):
            n_k.w[i] += eta * delta_w[i]

    for j in range(len(nn.hidden_layer)):
        n_j = nn.hidden_layer[j]
        delta_w = []
        for i in range(len(nn.input)):
            o_i = nn.input[i]
            delta_w.append(delta_j[j] * o_i)
        for i in range(len(delta_w)):
            n_j.w[i] += eta * delta_w[i]


class NeuralNetwork:
    def __init__(
        self,
        activation: ActivationFunction,
        architecture: Architecture,
        learning_algorithm: Callable = back_propagation
    ):
        self.activation = activation
        self.learning_algorithm = learning_algorithm

        self.hidden_layer = NeuronLayer(
                size=architecture.number_hidden,
                activation=activation,
                weights=architecture.hidden_weights,
                bias=architecture.hidden_bias)
        self.output_layer = NeuronLayer(
                size=architecture.number_outputs,
                activation=activation,
                weights=architecture.output_weights,
                bias=architecture.output_bias)

        self.input: Sequence[float]
        self.out: Sequence[float]

    # feed-forward william carmine
    def __call__(self, *args: float) -> Sequence[float]:
        self.input = tuple(args)
        self.out = self.output_layer(*self.hidden_layer(*self.input))
        return self.out

    def train(
        self,
        patterns: Sequence[Tuple[Sequence[float], Sequence[float]]],
        epoch_number: int,
        eta: float = 0.5
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
            self(*x)
            for i in range(len(d)):
                error += 0.5 * (d[i] - self.out[i])**2
        return error

    class Architecture:
        def __init__(
            self,
            number_inputs: int,
            number_hidden: int,
            number_outputs: int,

            hidden_weights: Optional[Sequence[List[float]]] = None,
            output_weights: Optional[Sequence[List[float]]] = None,

            hidden_bias: Optional[float] = None,
            output_bias: Optional[float] = None,

            threshold: Optional[int] = None,
            range_weights: Optional[float] = None,

        ):

            self.number_inputs: int = number_inputs
            self.number_hidden: int = number_hidden
            self.number_outputs: int = number_outputs

            self.threshold: int = threshold
            self.range_weights: float = range_weights

            self.hidden_bias: float
            self.output_bias: float

            self.hidden_weights: Sequence[List[float]]
            self.output_weights: Sequence[List[float]]

            if hidden_weights is None:
                _hidden_weights = []
                for _ in range(self.number_hidden):
                    w = []
                    for _ in range(self.number_inputs):
                        # starting values updated:
                        # if #inputs > threshold ==> as before: rand * 0.4 - 0.2
                        # else #inputs <= threshold ==> (rand * 0.4 - 0.2) * 2 / #inputs

                        w.append(random.uniform(- self.range_weights, self.range_weights)
                                 if self.number_inputs > self.threshold else
                                 random.uniform(- self.range_weights, self.range_weights) * 2 / self.number_inputs)
                    _hidden_weights.append(w)
                self.hidden_weights = _hidden_weights
            else:
                self.hidden_weights = hidden_weights

            if output_weights is None:
                _output_weights = []
                for _ in range(self.number_outputs):
                    w = []
                    for _ in range(self.number_hidden):
                        w.append(random.uniform(- self.range_weights, self.range_weights)
                                 if self.number_inputs > self.threshold else
                                 random.uniform(- self.range_weights, self.range_weights) * 2 / self.number_inputs)
                    _output_weights.append(w)
                self.output_weights = _output_weights
            else:
                self.output_weights = output_weights

            if hidden_bias is None:
                self.hidden_bias = random.uniform(- self.range_weights, self.range_weights) \
                    if self.number_inputs > self.threshold else \
                    random.uniform(- self.range_weights, self.range_weights) * 2 / self.number_inputs
            else:
                self.hidden_bias = hidden_bias

            if output_bias is None:
                self.output_bias = random.uniform(- self.range_weights, self.range_weights) \
                                 if self.number_inputs > self.threshold else \
                                 random.uniform(- self.range_weights, self.range_weights) * 2 / self.number_inputs
            else:
                self.output_bias = output_bias
