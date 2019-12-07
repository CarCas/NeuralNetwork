from typing import Sequence
import numpy as np
from scipy.special import expit  # type: ignore

from nn.types import BaseNeuralNetwork, Pattern, ActivationFunction


def generate_layer(size_layer: int, size_previous_layer: int):
    range_weights = 1/np.sqrt(size_previous_layer)
    nodes_rand = np.random.uniform(-range_weights, range_weights, (size_layer, size_previous_layer))
    return np.insert(nodes_rand, 0, 0, 1)


def fprime(out: float):
    return out * (1 - out)


class MLPMatrix(BaseNeuralNetwork):
    def __init__(
        self,
        *layer_sizes: int,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction,
        eta: float
    ):
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(generate_layer(layer_sizes[i], layer_sizes[i-1]))
        self.layers = np.array(layers)

        self.activation: ActivationFunction = activation
        self.activation_hidden: ActivationFunction = activation_hidden
        self.eta: float = eta

    def __call__(self, *input: Sequence[float]) -> Sequence[Sequence[float]]:
        self.inputs = [np.insert(input, 0, 1, 1)]
        self.outputs = []
        for layer in self.layers[:-1]:
            self.outputs.append(self.activation_hidden(self.inputs[-1] @ layer.T))
            self.inputs.append(np.insert(self.outputs[-1], 0, 1, 1))
        self.outputs.append(self.activation(self.inputs[-1] @ self.layers[-1].T))
        return self.outputs[-1]

    def train(self, patterns: Sequence[Pattern]) -> None:
        x, d = zip(*patterns)
        self(*x)

        delta = [(d - self.outputs[-1]) * self.activation.derivative(self.outputs[-1])]
        for i in range(len(self.layers))[::-1][1:]:
            delta.insert(0, (
                np.dot(self.layers[i+1].T[1:], delta[0].T).T * self.activation_hidden.derivative(self.outputs[i])))

        self.gradients = []
        for i in range(len(self.layers)):
            self.gradients.append(np.mean((delta[i] * self.inputs[i][np.newaxis].T).T, axis=1))
            self.layers[i] += self.eta * self.gradients[i]
