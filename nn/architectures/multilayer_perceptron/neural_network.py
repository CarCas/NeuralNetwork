from typing import Sequence
import numpy as np

from nn.types import Pattern
from nn.activation_function import ActivationFunction

from nn.architectures.multilayer_perceptron.weights_generator import WeightsGenerator
from nn.architectures.multilayer_perceptron.neuron_layer import NeuronLayer
from nn.architectures.multilayer_perceptron.learning_algorithms import LeariningAlgorthm
from nn.architectures.multilayer_perceptron.types import MLPBaseNeuralNetwork


class MLPNeuralNetwork(MLPBaseNeuralNetwork):
    '''
    Instean of instantiate directly this class, use
    nn.architecures.multiplayer_perceptron.architecyre.Architecture
    '''
    def __init__(
        self,
        activation: ActivationFunction,
        activation_hidden: ActivationFunction,
        eta: float,
        learining_algorthm: LeariningAlgorthm,
        weights_generator: WeightsGenerator,
    ) -> None:
        self.activation: ActivationFunction = activation
        self.activation_hidden: ActivationFunction = activation_hidden
        self.eta: float = eta
        self.learining_algorthm: LeariningAlgorthm = learining_algorthm

        self._hidden_layer: NeuronLayer = NeuronLayer(
                activation=activation_hidden,
                weights=weights_generator.hidden_weights)
        self._output_layer: NeuronLayer = NeuronLayer(
                activation=activation,
                weights=weights_generator.output_weights)

        self._input: Sequence[float] = np.zeros(weights_generator.size_input_nodes)
        self._out: Sequence[float] = []

    def __call__(self, *args: float) -> Sequence[float]:
        self._input = args
        self._out = self.output_layer(*self.hidden_layer(*self.input))
        return self._out

    def train(self, patterns: Sequence[Pattern]) -> None:
        self.learining_algorthm(patterns, self, self.eta)

    @property
    def out(self) -> Sequence[float]:
        return self._out

    @property
    def input(self) -> Sequence[float]:
        return self._input

    @property
    def output_layer(self) -> NeuronLayer:
        return self._output_layer

    @property
    def hidden_layer(self) -> NeuronLayer:
        return self._hidden_layer
