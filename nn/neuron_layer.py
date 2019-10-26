from typing import Optional, Sequence, List

from nn.neuron import Neuron
from nn.activation_function import ActivationFunction


class NeuronLayer:
    def __init__(
        self,
        size: int,
        activation: ActivationFunction,
        weights: Optional[Sequence[List[float]]] = None,
        bias: float = 0,
    ):
        self.neurons: Sequence[Neuron] = tuple(
            Neuron(activation, weights[i], bias) for i in range(size))
        self.out: Sequence[float]

    def __call__(self, *args: float) -> Sequence[float]:
        self.out = tuple(neuron(*args) for neuron in self.neurons)
        return self.out

    def w_from(self, index: int) -> Sequence[float]:
        return tuple(neuron.w[index] for neuron in self.neurons)

    def __len__(self) -> int:
        return len(self.neurons)

    def __getitem__(self, i) -> Neuron:
        return self.neurons[i]
