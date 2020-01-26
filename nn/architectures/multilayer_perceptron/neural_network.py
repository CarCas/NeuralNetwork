from typing import Sequence, Optional
import numpy as np

from nn.types import BaseNeuralNetwork, Pattern, ActivationFunction


def _generate_layer(size_layer: int, size_previous_layer: int) -> Sequence[Sequence[float]]:
    range_weights = 1/np.sqrt(size_previous_layer)
    nodes_rand = np.random.uniform(-range_weights, range_weights, (size_layer, size_previous_layer))
    return np.insert(nodes_rand, 0, 0, 1)


class MLPNeuralNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        size_hidden_layers: Sequence[int],
        activation: ActivationFunction,
        activation_hidden: ActivationFunction,
        eta: float,
        alpha: float,
        alambd: float,

        eta_decay: float,

        layers: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    ):
        self.size_hidden_layers = size_hidden_layers
        self.activation: ActivationFunction = activation
        self.activation_hidden: ActivationFunction = activation_hidden
        self.eta: float = eta
        self.alpha: float = alpha
        self.alambd: float = alambd

        self.eta_decay: float = eta_decay
        self.eta_min: float = self.eta / 100
        self.iterations: int = 0

        self.len_layers = len(size_hidden_layers) + 1

        self._are_layers_init = False
        self.layers = [np.array(0) for _ in range(self.len_layers)]

        if layers is not None:
            self._are_layers_init = True
            self.layers = [np.array(layer) for layer in layers]
            self.len_layers = len(layers)

        self._inputs = [np.array(0) for _ in range(self.len_layers)]
        self._outputs = [np.array(0) for _ in range(self.len_layers)]
        self._deltas = [np.array(0) for _ in range(self.len_layers)]
        self._gradients = [np.array(0) for _ in range(self.len_layers)]
        self._delta_w = [np.array(0) for _ in range(self.len_layers)]

    def __call__(self, *inp: Sequence[float]) -> Sequence[Sequence[float]]:
        if not self._are_layers_init:
            raise RuntimeError('Network not fitted')

        layers = self.layers
        inputs = self._inputs
        outputs = self._outputs
        len_layers = self.len_layers
        activation_hidden = self.activation_hidden

        inputs[0] = np.insert(inp, 0, 1, 1)
        for i in range(len_layers-1):
            outputs[i] = activation_hidden(inputs[i] @ layers[i].T)
            inputs[i+1] = np.insert(outputs[i], 0, 1, 1)
        outputs[-1] = self.activation(inputs[-1] @ layers[-1].T)
        return outputs[-1]

    def fit(self, patterns: Sequence[Pattern]) -> None:
        if not self._are_layers_init:
            self._compute_layers(len(patterns[0][0]), len(patterns[0][1]))

        layers = self.layers
        inputs = self._inputs
        outputs = self._outputs
        deltas = self._deltas
        gradients = self._gradients
        delta_w = self._delta_w
        activation_hidden = self.activation_hidden
        eta = max(self.eta_min, self.eta * (1. / (1. + self.eta_decay * self.iterations)))
        alpha = self.alpha
        alambd = self.alambd

        x: Sequence[Sequence[float]]
        d: Sequence[Sequence[float]]
        x, d = zip(*patterns)
        self(*x)

        deltas[-1] = (d - outputs[-1]) * self.activation.derivative(outputs[-1])
        for i in range(self.len_layers)[-2::-1]:
            deltas[i] = np.multiply(np.dot(layers[i+1].T[1:],
                                           deltas[i+1].T).T,
                                    activation_hidden.derivative(outputs[i]))

        for i in range(self.len_layers):
            gradients[i] = np.mean((deltas[i] * inputs[i][np.newaxis].T).T, axis=1)
            delta_w[i] = eta * gradients[i] + alpha * delta_w[i]
            layers[i] += delta_w[i] - alambd * layers[i]

        self.iterations += 1

    def _compute_layers(self, input_size: int, output_size: int) -> None:
        size_layers = list(self.size_hidden_layers)

        size_layers = [input_size] + size_layers + [output_size]

        tmp_layers = [_generate_layer(size_layers[i], size_layers[i-1])
                      for i in range(1, len(size_layers))]

        layers = [np.array(layer) for layer in tmp_layers]

        self._are_layers_init, self.layers = True, layers

    @property
    def weights(self) -> Sequence[Sequence[Sequence[float]]]:
        if not self._are_layers_init:
            raise RuntimeError('Network not fitted')

        return self.layers

    @property
    def inputs(self) -> Sequence[Sequence[Sequence[float]]]:
        return self._inputs

    @property
    def outputs(self) -> Sequence[Sequence[Sequence[float]]]:
        return self._outputs

    @property
    def gradients(self) -> Sequence[Sequence[Sequence[float]]]:
        return self._gradients
