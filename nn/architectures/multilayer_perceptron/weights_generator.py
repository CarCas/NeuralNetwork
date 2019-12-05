from typing import Optional, Sequence
import numpy as np

from nn.architectures.multilayer_perceptron.types import NeuronWeights, LayerWeights


class WeightsGenerator:
    def __init__(
        self,

        size_input_layer: int,
        sizes_hidden_layers: Sequence[int],
        size_output_layer: int,

        hidden_weights: Optional[Sequence[LayerWeights]] = None,
        output_weights: Optional[LayerWeights] = None,

        range_weights: Optional[float] = None,
    ):
        self.size_input_layer: int = size_input_layer
        self.sizes_hidden_layers: Sequence[int] = sizes_hidden_layers
        self.size_output_layer: int = size_output_layer

        self.range_weights: Optional[float] = range_weights

        self._hidden_weights: Optional[Sequence[LayerWeights]] = hidden_weights
        self._output_weights: Optional[LayerWeights] = output_weights

    @property
    def hidden_weights(self) -> Sequence[LayerWeights]:
        if self._hidden_weights is not None:
            return self._hidden_weights
        hidden_layers_weights = [self(self.sizes_hidden_layers[0], self.size_input_layer)]
        for i in range(1, len(self.sizes_hidden_layers)):
            hidden_layers_weights.append(self(self.sizes_hidden_layers[i], self.sizes_hidden_layers[i-1]))
        return hidden_layers_weights

    @property
    def output_weights(self) -> LayerWeights:
        if self._output_weights is not None:
            return self._output_weights
        return self(self.size_output_layer, self.sizes_hidden_layers[-1])

    def __call__(self, number_nodes: int, size_weights: int) -> Sequence[Sequence[float]]:
        if self.range_weights is None:
            # Glorot, Bengio AISTATS 2010
            range_weights = 1/np.sqrt(size_weights)
            nodes_rand = np.random.uniform(-range_weights, range_weights, (number_nodes, size_weights))
            nodes = np.zeros((number_nodes, size_weights+1))
            nodes[:, 1:] = nodes_rand
        else:
            nodes = np.random.uniform(-self.range_weights, self.range_weights, (number_nodes, 1+size_weights))
        return nodes
