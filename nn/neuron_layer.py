from typing import Optional, Sequence

from nn import Neuron, ActivationFunction


class NeuronLayer:
    def __init__(
        self,
        size: int,
        activation: ActivationFunction,
        weights: Optional[Sequence[Sequence[float]]] = None,
        bias: float = 0,
    ):
        self._neurons = tuple(
            Neuron(activation, weights[i], bias)
            for i in range(size))
        self._out = None

    def __call__(self, *args: float) -> Sequence[float]:
        self._out = tuple(neuron(*args) for neuron in self.neurons)
        return self.out

    def w_from(self, index: int) -> Sequence[float]:
        return tuple(neuron.w[index] for neuron in self.neurons)

    @property
    def out(self) -> Sequence[float]:
        if self._out is None:
            raise RuntimeError('neural network not executed')
        return self._out

    @property
    def neurons(self) -> Sequence[Neuron]:
        return self._neurons

    def __len__(self) -> int:
        return len(self._neurons)

    def __getitem__(self, i) -> Neuron:
        return self._neurons[i]
