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

        range_weights: number = 0.7,
        threshold: Optional[int] = None,
    ):
        self.size_input_nodes: int = size_input_nodes
        self.size_hidden_nodes: int = size_hidden_nodes
        self.size_output_nodes: int = size_output_nodes

        self.threshold: Optional[int] = threshold
        self.range_weights: number = range_weights

        self.hidden_weights: Sequence[Sequence[number]]
        self.output_weights: Sequence[Sequence[number]]

        if hidden_weights is None:
            self.hidden_weights = self.generate_random_layer(self.size_hidden_nodes, self.size_input_nodes)
        else:
            self.hidden_weights = hidden_weights

        if output_weights is None:
            self.output_weights = self.generate_random_layer(self.size_output_nodes, self.size_hidden_nodes)
        else:
            self.output_weights = output_weights

    def generate_random_layer(self, number_nodes: int, size_weights: int) -> Sequence[Sequence[number]]:
        nodes = np.random.uniform(-self.range_weights, self.range_weights, (number_nodes, 1+size_weights))
        if self.threshold is None or size_weights <= self.threshold:
            nodes *= 2 / size_weights
        return nodes
