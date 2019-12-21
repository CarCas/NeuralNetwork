from typing import Sequence, Optional
import numpy as np

from nn.types import Architecture as BaseArchitecure, ActivationFunction
from nn.architectures.multilayer_perceptron.neural_network import MLPNeuralNetwork
from nn.activation_functions import sigmoid


def generate_layer(size_layer: int, size_previous_layer: int) -> Sequence[Sequence[float]]:
    range_weights = 1/np.sqrt(size_previous_layer)
    nodes_rand = np.random.uniform(-range_weights, range_weights, (size_layer, size_previous_layer))
    return np.insert(nodes_rand, 0, 0, 1)


class MultilayerPerceptron(BaseArchitecure):
    def __init__(
        self,
        *layer_sizes: int,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction = sigmoid,
        eta: float = 0.5,
        alpha: float = 0,

        layers: Optional[Sequence[Sequence[Sequence[float]]]] = None
    ) -> None:
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.eta = eta
        self.alpha = alpha

        self.layer_sizes: Sequence[int] = layer_sizes
        self.layers: Optional[Sequence[Sequence[Sequence[float]]]] = layers

    def __call__(self) -> MLPNeuralNetwork:
        if self.layers is None:
            layers = [generate_layer(self.layer_sizes[i], self.layer_sizes[i-1])
                      for i in range(1, len(self.layer_sizes))]
        else:
            layers = self.layers

        return MLPNeuralNetwork(
            layers=layers,
            activation=self.activation,
            activation_hidden=self.activation_hidden,
            eta=self.eta,
            alpha=self.alpha,
        )
