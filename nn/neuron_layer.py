from typing import Sequence
import numpy as np

from nn.neuron import Neuron, ActivationFunction


class NeuronLayer:
    def __init__(
        self,
        weights: Sequence[Sequence[float]],
        activation: ActivationFunction,
    ):
        self.activation = activation
        self.neurons: Sequence[Neuron] = [Neuron(activation=activation, w=w) for w in weights]

        self.out: Sequence[float]
        self.fprime: Sequence[float]

    def __call__(self, *args: float) -> Sequence[float]:
        self.out = tuple(neuron(*args) for neuron in self.neurons)
        self.fprime = tuple(n.fprime for n in self.neurons)
        return self.out

    @property
    def w(self) -> Sequence[Sequence[float]]:
        return np.array([n.w for n in self.neurons])

    @w.setter
    def w(self, w: Sequence[Sequence[float]]) -> None:
        for i, n in enumerate(self.neurons):
            n.w = w[i]

    def __len__(self) -> int:
        return len(self.neurons)
