from __future__ import annotations

from typing import Sequence, MutableSequence, Optional, Dict, Any, Tuple, TypeVar
import numpy as np
from copy import deepcopy

from nn.types import Pattern, Architecture, BaseNeuralNetwork, ActivationFunction
from nn.learning_algorithm import LearningAlgorithm, batch
from nn.activation_functions import sigmoid
from nn.error_calculator import ErrorCalculator


T = TypeVar('T')
Container = MutableSequence[T]


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
    _internal_networks: contains a neural network for each epoch of training,
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
        epochs_limit: int = 1,
        epsilon: float = -1,
        n_init: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        self.architecture = architecture
        self.error_calculator: ErrorCalculator = error_calculator
        self.learning_algorithm: LearningAlgorithm = learning_algorithm
        self.epochs_limit: int = epochs_limit
        self.epsilon: float = epsilon
        self.n_init: int = n_init
        if seed is not None:
            np.random.seed(seed)

        self._current_network: BaseNeuralNetwork = architecture()
        self._internal_networks: MutableSequence[BaseNeuralNetwork] = []

    def set(self, **kwargs) -> NeuralNetwork:
        self.__dict__.update(**kwargs)
        self.__init__(**self.__dict__)  # type: ignore
        return self

    # feed-forward
    def __call__(self, *args: Sequence[float]) -> Sequence[Sequence[float]]:
        return self._current_network(*args)

    def fit(
        self,
        patterns: Sequence[Pattern],
    ) -> None:
        container_best_trained_network: Container[Tuple[float, Optional[NeuralNetwork]]] = [(float("inf"), None)]

        for _ in range(self.n_init):
            for _ in range(self.epochs_limit):
                self.learning_algorithm(self._current_network, patterns)
                self._update_internal_networks()
                if self._early_stopping():
                    break

            self._update_best_trained_network(container_best_trained_network, patterns)
        self._fetch_best_trained_network(container_best_trained_network[0])

    def compute_error(self, patterns: Sequence[Pattern], error_calculator: ErrorCalculator = None) -> float:
        error_calculator = self.error_calculator if error_calculator is None else error_calculator
        return self.error_calculator([self], patterns)[0]

    def compute_learning_curve(self, patterns: Sequence[Pattern], error_calculator: ErrorCalculator = None) -> Sequence[float]:
        error_calculator = self.error_calculator if error_calculator is None else error_calculator
        return self.error_calculator(self._internal_networks, patterns)

    def _update_internal_networks(self) -> None:
        self._internal_networks.append(deepcopy(self._current_network))

    def _update_best_trained_network(
        self,
        container_best_network: Container[Tuple[float, Optional[NeuralNetwork]]],
        patterns: Sequence[Pattern]
    ) -> None:
        score = self.compute_error(patterns, ErrorCalculator.MSE)
        if score < container_best_network[0][0]:
            container_best_network[0] = score, deepcopy(self)
        self.set()

    def _fetch_best_trained_network(self, best_network: Tuple[float, Optional[NeuralNetwork]]):
        if best_network[1] is not None:
            self._current_network = best_network[1]._current_network
            self._internal_networks = best_network[1]._internal_networks

    def _early_stopping(self) -> bool:
        if self.epsilon >= 0:
            l2_gradient = np.sqrt(np.sum([np.sum(np.square(x)) for x in self._current_network.gradients]))
            return l2_gradient <= self.epsilon
        return False

    @property
    def weights(self) -> Sequence[Sequence[Sequence[float]]]:
        return self._current_network.weights

    @property
    def inputs(self) -> Sequence[Sequence[Sequence[float]]]:
        return self._current_network.inputs

    @property
    def outputs(self) -> Sequence[Sequence[Sequence[float]]]:
        return self._current_network.outputs

    @property
    def gradients(self) -> Sequence[Sequence[Sequence[float]]]:
        return self._current_network.gradients
