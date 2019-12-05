from typing import Sequence, Optional

from nn.types import Architecture as BaseArchitecure
from nn.activation_function import ActivationFunction, sigmoid

from nn.config import DEFAULT_ETA

from nn.architectures.multilayer_perceptron.types import LayerWeights
from nn.architectures.multilayer_perceptron.weights_generator import WeightsGenerator
from nn.architectures.multilayer_perceptron.learning_algorithms import (
    LeariningAlgorthm,
    Batch,
)

from nn.architectures.multilayer_perceptron.neural_network import MLPNeuralNetwork


class MultilayerPerceptron(BaseArchitecure):
    '''
    size_input_layer: number of inputs
    sizes_hidden_layers: sequence containing the number of each hidden nodes;
                         e.g.: for a single hidden layer use a list with just one int
    size_output_layer: number of output nodes

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
        size_input_layer: int,
        sizes_hidden_layers: Sequence[int],
        size_output_layer: int,

        learning_algorithm: LeariningAlgorthm = Batch(),

        range_weights: Optional[float] = None,

        hidden_weights: Optional[Sequence[LayerWeights]] = None,
        output_weights: Optional[LayerWeights] = None,
    ):
        self.learning_algorithm: LeariningAlgorthm = learning_algorithm

        self.weights_generator: WeightsGenerator = WeightsGenerator(
            size_input_layer=size_input_layer,
            sizes_hidden_layers=sizes_hidden_layers,
            size_output_layer=size_output_layer,
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
            learning_algorithm=self.learning_algorithm,
            weights_generator=self.weights_generator,
        )
