from typing import Sequence, Optional

from nn.types import Architecture as BaseArchitecure, ActivationFunction

from nn.architectures.multilayer_perceptron.neural_network import MLPMatrix


class MultilayerPerceptron(BaseArchitecure):
    def __init__(self, *layer_sizes: int):
        self.layer_sizes: Sequence[int] = layer_sizes

    def __call__(
        self,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction,
        eta: float,
    ) -> MLPMatrix:
        return MLPMatrix(
            *self.layer_sizes,
            activation=activation,
            activation_hidden=activation_hidden,
            eta=eta
        )
