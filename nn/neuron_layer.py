from typing import Sequence
import numpy as np

from nn.neuron import Neuron
from nn.activation_function import ActivationFunction
from nn.number import number


class NeuronLayer:
    def __init__(
        self,
        weights: Sequence[Sequence[number]],
        activation: ActivationFunction,
    ):
        self.activation = activation
        self.neurons: Sequence[Neuron] = (
            np.vectorize(lambda w: Neuron(activation=activation, w=w), signature='(n)->()'
                         )(np.array(weights)))

        self.out: Sequence[number]
        self.fprime: Sequence[number]

    def __call__(self, *args: number) -> Sequence[number]:
        self.out = np.vectorize(lambda neuron: neuron(*args))(self.neurons)
        self.fprime = np.vectorize(lambda neuron: neuron.fprime)(self.neurons)
        return self.out

    @property
    def w(self) -> Sequence[Sequence[number]]:
        return np.vectorize(lambda neuron: neuron.w, signature='()->(n)')(self.neurons)

    @w.setter
    def w(self, weights: Sequence[Sequence[number]]) -> None:
        self.neurons: Sequence[Neuron] = (
            np.vectorize(lambda w: Neuron(activation=self.activation, w=w), signature='(n)->()'
                         )(np.array(weights)))

    def __len__(self) -> int:
        return len(self.neurons)

    def __getitem__(self, i) -> Neuron:
        return self.neurons[i]
