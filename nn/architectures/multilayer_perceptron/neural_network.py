from typing import Sequence
import numpy as np
from scipy.special import expit  # type: ignore

from nn.types import BaseNeuralNetwork, Pattern, ActivationFunction


class MLPNeuralNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        layers: Sequence[Sequence[Sequence[float]]],
        activation: ActivationFunction,
        activation_hidden: ActivationFunction,
        eta: float
    ):
        self.layers = [np.array(layer) for layer in layers]
        self.activation: ActivationFunction = activation
        self.activation_hidden: ActivationFunction = activation_hidden
        self.eta: float = eta

        self.len_layers = len(layers)
        self.inputs = [np.array(0)] * self.len_layers
        self.outputs = [np.array(0)] * self.len_layers

        self.deltas = [np.array(0)] * self.len_layers
        self.gradients = [np.array(0)] * self.len_layers

    def __call__(self, *input: Sequence[float]) -> Sequence[Sequence[float]]:
        layers = self.layers
        inputs = self.inputs
        outputs = self.outputs
        len_layers = self.len_layers
        activation_hidden = self.activation_hidden

        inputs[0] = np.insert(input, 0, 1, 1)
        for i in range(len_layers-1):
            outputs[i] = activation_hidden(inputs[i] @ layers[i].T)
            inputs[i+1] = np.insert(outputs[i], 0, 1, 1)
        outputs[-1] = self.activation(inputs[-1] @ layers[-1].T)
        return outputs[-1]

    def train(self, patterns: Sequence[Pattern]) -> None:
        layers = self.layers
        inputs = self.inputs
        outputs = self.outputs
        deltas = self.deltas
        gradients = self.gradients
        activation_hidden = self.activation_hidden
        eta = self.eta

        x, d = zip(*patterns)
        self(*x)

        deltas[-1] = (d - outputs[-1]) * self.activation.derivative(outputs[-1])
        for i in range(self.len_layers)[-2::-1]:
            deltas[i] = np.multiply(np.matmul(layers[i+1].T[1:],
                                              deltas[i+1].T).T,
                                    activation_hidden.derivative(outputs[i]))

        for i in range(self.len_layers):
            gradients[i] = np.mean((deltas[i] * inputs[i][np.newaxis].T).T, axis=1)
            layers[i] += eta * gradients[i]
