from __future__ import annotations

from typing import Sequence, MutableSequence, Optional
import numpy as np
from copy import deepcopy

from nn.types import Pattern, Architecture, BaseNeuralNetwork, ActivationFunction
from nn.learning_algorithm import LearningAlgorithm, batch
from nn.activation_functions import sigmoid
from nn.error_calculator import ErrorCalculator


class NeuralNetwork(BaseNeuralNetwork):
    """
    Params and istance variables
    ----------------------------------------
    activation: activation function of output layer
    architecture: sizes of the network

    activation_hidden: activation function of hidden layers
    error_calculator: to specify witch kind of errors the nn produces

    learning_algorithm: callable that train the network given network and patterns
    eta: learning rate
    epochs_limit: max possible float of epoch during a train call
    epsilon: TODO

    Istance variables
    ----------------------------------------
    learning_networks: contains a neural network for each epoch of training,
                       used to compute errors and learning curve

    Debug params
    ----------------------------------------
    seed: to set a random seed

    Public methods
    ----------------------------------------
    __call__: execute feed forward
    set: reset the network, if some parameters are passed their are setted in the network
    train: train the network, fill `training_errors` and `testing_errors` instance variable
    compute_error: compute the error for the patterns of the specified error_generator
    compute_learning_curve: compute an error for each nn in learning_network
    """
    def __init__(
        self,

        architecture: Architecture,

        error_calculator: ErrorCalculator = ErrorCalculator.MSE,

        learning_algorithm: LearningAlgorithm = batch,

        penalty: float = 0.1,

        epochs_limit: int = 1,
        epsilon: float = -1,

        dim_mini_batch: int = -1,

        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.architecture = architecture
        self.internal_network: BaseNeuralNetwork = architecture()

        self.learning_algorithm: LearningAlgorithm = learning_algorithm
        self.error_calculator: ErrorCalculator = error_calculator

        self.epsilon: float = epsilon
        self.epochs_limit: int = epochs_limit

        self.penalty: float = penalty
        self.dim_mini_batch = dim_mini_batch

        self.learning_networks: MutableSequence[BaseNeuralNetwork] = []

    def set(self, **kwargs) -> NeuralNetwork:
        self.__dict__.update(**kwargs)
        self.__init__(**self.__dict__)
        return self

    # feed-forward
    def __call__(self, *args: Sequence[float]) -> Sequence[Sequence[float]]:
        return self.internal_network(*args)

    def train(
        self,
        patterns: Sequence[Pattern],
    ) -> None:
        for _ in range(self.epochs_limit):
            self.learning_algorithm(self.internal_network, patterns)
            self._update_training_networks()
            if self._early_stopping():
                break

    def compute_error(self, patterns: Sequence[Pattern]) -> float:
        return self.error_calculator([self], patterns)[0]

    def compute_learning_curve(self, patterns: Sequence[Pattern]) -> Sequence[float]:
        return self.error_calculator(self.learning_networks, patterns)

    def _update_training_networks(self) -> None:
        return self.learning_networks.append(deepcopy(self.internal_network))

    def _early_stopping(self) -> bool:
        if self.epsilon >= 0:
            l2_gradient = np.sqrt(np.sum([np.sum(np.square(x)) for x in self.internal_network.gradients]))
            return l2_gradient <= self.epsilon
        return False

    @property
    def weights(self) -> Sequence[Sequence[Sequence[float]]]:
        return self.internal_network.weights

    @property
    def inputs(self) -> Sequence[Sequence[Sequence[float]]]:
        return self.internal_network.inputs

    @property
    def outputs(self) -> Sequence[Sequence[Sequence[float]]]:
        return self.internal_network.outputs

    @property
    def gradients(self) -> Sequence[Sequence[Sequence[float]]]:
        return self.internal_network.gradients
