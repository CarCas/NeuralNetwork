import abc
from typing import Sequence

from nn.types import BaseNeuralNetwork, NeuronWeights
from nn.architectures.multilayer_perceptron.neuron_layer import NeuronLayer


LayerWeights = Sequence[NeuronWeights]


class MLPBaseNeuralNetwork(BaseNeuralNetwork):
    @property
    @abc.abstractmethod
    def output_layer(self) -> NeuronLayer:
        pass

    @property
    @abc.abstractmethod
    def hidden_layers(self) -> Sequence[NeuronLayer]:
        pass
