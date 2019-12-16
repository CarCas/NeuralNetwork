from typing import Sequence, Optional
import numpy as np

from nn.types import Architecture as BaseArchitecure, ActivationFunction
from nn.architectures.multilayer_perceptron.neural_network import MLPNeuralNetwork


def generate_layer(size_layer: int, size_previous_layer: int) -> Sequence[Sequence[float]]:
    range_weights = 1/np.sqrt(size_previous_layer)
    nodes_rand = np.random.uniform(-range_weights, range_weights, (size_layer, size_previous_layer))
    return np.insert(nodes_rand, 0, 0, 1)


class MultilayerPerceptron(BaseArchitecure):
    def __init__(
        self,
        *layer_sizes: int,
        layers: Optional[Sequence[Sequence[Sequence[float]]]] = None
    ) -> None:
        self.layer_sizes: Sequence[int] = layer_sizes
        self.layers: Optional[Sequence[Sequence[Sequence[float]]]] = layers

    def __call__(
        self,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction,
        eta: float,
        alpha: float
    ) -> MLPNeuralNetwork:
        if self.layers is None:
            layers = [generate_layer(self.layer_sizes[i], self.layer_sizes[i-1])
                      for i in range(1, len(self.layer_sizes))]
        else:
            layers = self.layers

        return MLPNeuralNetwork(
            layers=layers,
            activation=activation,
            activation_hidden=activation_hidden,
            eta=eta,
            alpha=alpha,
        )
