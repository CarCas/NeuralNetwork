from typing import Sequence, Optional
import numpy as np

from nn.types import Architecture as BaseArchitecure, ActivationFunction
from nn.architectures.multilayer_perceptron.neural_network import MLPNeuralNetwork
from nn.activation_functions import sigmoid


class MultilayerPerceptron(BaseArchitecure):
    def __init__(
        self,
        *size_hidden_layers: int,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction = sigmoid,
        eta: float = 0.001,
        alpha: float = 0.9,
        alambd: float = 0.0001,

        layers: Optional[Sequence[Sequence[Sequence[float]]]] = None
    ) -> None:
        self.size_hidden_layers = size_hidden_layers
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.eta = eta
        self.alpha = alpha
        self.alambd = alambd

        self.layers = layers

    def __call__(self) -> MLPNeuralNetwork:
        return MLPNeuralNetwork(
            size_hidden_layers=self.size_hidden_layers,
            activation=self.activation,
            activation_hidden=self.activation_hidden,
            eta=self.eta,
            alpha=self.alpha,
            alambd=self.alambd,

            layers=self.layers,
        )
