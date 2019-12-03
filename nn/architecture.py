from typing import Optional, Sequence, Union
from nn.number import number
import numpy as np


class Architecture:
    def __init__(
        self,

        size_input_nodes: int,
        size_hidden_nodes: int,
        size_output_nodes: int,

        hidden_weights: Optional[Sequence[Sequence[number]]] = None,
        output_weights: Optional[Sequence[Sequence[number]]] = None,

        range_weights: Optional[number] = None,
        threshold: Optional[int] = None,
    ):
        self.size_input_nodes: int = size_input_nodes
        self.size_hidden_nodes: int = size_hidden_nodes
        self.size_output_nodes: int = size_output_nodes

        self.threshold: Optional[int] = threshold
        self.range_weights: Optional[number] = range_weights

        self._hidden_weights = hidden_weights
        self._output_weights = output_weights

    @property
    def hidden_weights(self) -> Sequence[Sequence[number]]:
        if self._hidden_weights is not None:
            return self._hidden_weights
        return self.generate_random_layer(self.size_hidden_nodes, self.size_input_nodes)

    @property
    def output_weights(self) -> Sequence[Sequence[number]]:
        if self._output_weights is not None:
            return self._output_weights
        return self.generate_random_layer(self.size_output_nodes, self.size_hidden_nodes)

    def generate_random_layer(self, number_nodes: int, size_weights: int) -> Sequence[Sequence[number]]:
        if self.range_weights is None:
            # Glorot, Bengio AISTATS 2010
            range_weights = 1/np.sqrt(size_weights)
            nodes_rand = np.random.uniform(-range_weights, range_weights, (number_nodes, size_weights))
            nodes = np.zeros((number_nodes, size_weights+1))
            nodes[:, 1:] = nodes_rand
        else:
            nodes = np.random.uniform(-self.range_weights, self.range_weights, (number_nodes, 1+size_weights))
            if self.threshold is None or size_weights <= self.threshold:
                nodes *= 2 / size_weights
        return nodes
