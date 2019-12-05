import abc

from nn.types import NeuralNetwork as BaseNeuralNetwork
from nn.architectures.multilayer_perceptron.neuron_layer import NeuronLayer


class NeuralNetwork(BaseNeuralNetwork):
    @abc.abstractmethod
    def output_layer(self) -> NeuronLayer:
        pass

    @abc.abstractmethod
    def hidden_layer(self) -> NeuronLayer:
        pass
