from typing import Sequence, Optional

from nn.types import Architecture as BaseArchitecure
from nn.activation_function import ActivationFunction, sigmoid

from nn.config import DEFAULT_ETA

from nn.architectures.multilayer_perceptron.weights_generator import WeightsGenerator
from nn.architectures.multilayer_perceptron.learning_algorithms import (
    LeariningAlgorthm,
    Batch,
)

from nn.architectures.multilayer_perceptron.neural_network import MLPNeuralNetwork


class MultilayerPerceptron(BaseArchitecure):
    '''
    size_input_nodes: number of inputs
    size_hidden_nodes: number of hidden nodes
    size_output_nodes: number of output nodes

    learning_algorithm: batch or online (from nn import Online, Batch)

    range_weights: if not specified weights are generated using Glorot-Bengio
                   otherwise weightsare are generated in range [-x, x]

    hidden_weights: if specified weights of hidden nodes are initialized with
                    the specified values
    output_weights: if specified weights of output nodes are initialized with
                    the specified values
    '''
    def __init__(
        self,
        size_input_nodes: int,
        size_hidden_nodes: int,
        size_output_nodes: int,

        learining_algorthm: LeariningAlgorthm = Batch(),

        range_weights: Optional[float] = None,
        threshold: Optional[int] = None,

        hidden_weights: Optional[Sequence[Sequence[float]]] = None,
        output_weights: Optional[Sequence[Sequence[float]]] = None,
    ):
        self.learining_algorthm: LeariningAlgorthm = learining_algorthm

        self.weights_generator: WeightsGenerator = WeightsGenerator(
            size_input_nodes=size_input_nodes,
            size_hidden_nodes=size_hidden_nodes,
            size_output_nodes=size_output_nodes,
            range_weights=range_weights,
            hidden_weights=hidden_weights,
            output_weights=output_weights,
        )

    def __call__(
        self,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction = sigmoid,
        eta: float = DEFAULT_ETA,
    ) -> MLPNeuralNetwork:
        return MLPNeuralNetwork(
            activation=activation,
            activation_hidden=activation_hidden,
            eta=eta,
            learining_algorthm=self.learining_algorthm,
            weights_generator=self.weights_generator,
        )
