from typing import Sequence, MutableSequence, Optional, Tuple, TypeVar
import numpy as np
from copy import deepcopy

from nn.types import Pattern, Architecture, BaseNeuralNetwork
from nn.learning_algorithm import LearningAlgorithm, batch
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
    epsilon: the threshold to set for the tolerance
    patience: the number of epochs under the epsilon value to wait before the training stops
    n_init: the number of reruns to compute in which the best model is chosen only depending on the training
    seed: the pseudo random number that initializes the weights of the nodes.

    Istance variables
    ----------------------------------------
    _internal_networks: contains a neural network for each epoch of training,
                       used to compute errors and learning curve

    Public methods
    ----------------------------------------
    __call__: execute feed forward
    set: reset the network, if some parameters are passed their are setted in the network

    __fit__: execute the training of the nn. If the validation_patterns or testing_patterns are passed,
    it saves the learning curves during training. If training_curve is True, then it saves internally the states
    of the network.

    compute_error: it computes the error on the trained network, given the patterns in input,
    if an ErrorCalculator is given, it is used instead of the default one

    compute_learning_curve: it computes all the learning curve on the trained network, given the patterns in input,
    if an ErrorCalculator is given, it is used instead of the default one.
    It returns an empty list if the learning curve has already been computed during the fit.

    the training_curve, validation_curve and testing_curve contain the respective learning curve,
    only if the learning curve has been computed during fit (check fit for more detail).

    """
    def __init__(
        self,
        architecture: Architecture,
        error_calculator: ErrorCalculator = ErrorCalculator.MSE,
        learning_algorithm: LearningAlgorithm = batch,
        epochs_limit: int = 1,
        epsilon: float = 1e-5,
        patience: int = 10,
        n_init: int = 1,

        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        self.architecture = architecture
        self.error_calculator: ErrorCalculator = error_calculator
        self.learning_algorithm: LearningAlgorithm = learning_algorithm
        self.epochs_limit: int = epochs_limit

        self.epsilon: float = epsilon
        self.patience: int = patience

        self.n_init: int = n_init

        self._last_gradient: int = 0
        self._current_patience: int = 0

        self._current_network: BaseNeuralNetwork = architecture()
        self._internal_networks: MutableSequence[BaseNeuralNetwork] = []
        self._errors: MutableSequence[float] = []

        self.training_curve: MutableSequence[float] = []
        self.validation_curve: MutableSequence[float] = []
        self.testing_curve: MutableSequence[float] = []

        self.seed = seed

    def set(self, **kwargs) -> 'NeuralNetwork':
        self.__dict__.update(**kwargs)
        self.__init__(**self.__dict__)  # type: ignore
        return self

    # feed-forward
    def __call__(self, *args: Sequence[float]) -> Sequence[Sequence[float]]:
        return self._current_network(*args)

    def fit(
        self,
        patterns: Sequence[Pattern],
        validation_patterns: Optional[Sequence[Pattern]] = None,
        testing_patterns: Optional[Sequence[Pattern]] = None,
        training_curve=True,
    ) -> None:
        container_best_trained_network: Container[Optional[Tuple[float, 'NeuralNetwork']]] = [
            None
        ]

        if self.seed is not None:
            np.random.seed(self.seed)

        for _ in range(self.n_init):
            for _ in range(self.epochs_limit):
                self.learning_algorithm(self._current_network, patterns)
                self._update_internal_networks(patterns, validation_patterns, testing_patterns, training_curve)
                if self._early_stopping():
                    break

            self._update_best_trained_network(container_best_trained_network, patterns)

        if container_best_trained_network[0] is not None:
            self._fetch_best_trained_network(container_best_trained_network[0])

    def compute_error(
            self,
            patterns: Sequence[Pattern],
            error_calculator: ErrorCalculator = None
    ) -> float:
        error_calculator = self.error_calculator if error_calculator is None else error_calculator
        return error_calculator([self], patterns)[0]

    def compute_learning_curve(
            self,
            patterns: Sequence[Pattern],
            error_calculator: ErrorCalculator = None
    ) -> Sequence[float]:
        error_calculator = self.error_calculator if error_calculator is None else error_calculator
        return error_calculator(self._internal_networks, patterns)

    def _update_internal_networks(
        self,
        patterns: Sequence[Pattern],
        validation_patterns: Optional[Sequence[Pattern]],
        testing_patterns: Optional[Sequence[Pattern]],
        training_curve: bool = True
    ) -> None:
        if validation_patterns is None:
            self._internal_networks.append(deepcopy(self._current_network))
        else:
            if training_curve:
                self.training_curve.append(self.compute_error(patterns))
            if len(validation_patterns):
                self.validation_curve.append(self.compute_error(validation_patterns))
            if testing_patterns is not None:
                self.testing_curve.append(self.compute_error(testing_patterns))

    def _update_best_trained_network(
        self,
        container_best_network: Container[Optional[Tuple[float, 'NeuralNetwork']]],
        patterns: Sequence[Pattern],
    ) -> None:
        score = self.compute_error(patterns)

        if container_best_network[0] is None \
                or self.error_calculator.choose((score, container_best_network[0][0]))[0] == 0:
            container_best_network[0] = score, self.copy()

        self.set()

    def _fetch_best_trained_network(self, best_network: Tuple[float, Optional['NeuralNetwork']]):
        if best_network[1] is not None:
            self._current_network = best_network[1]._current_network
            self._internal_networks = best_network[1]._internal_networks
            self._errors = best_network[1]._errors

            self.training_curve = best_network[1].training_curve
            self.validation_curve = best_network[1].validation_curve
            self.testing_curve = best_network[1].testing_curve

    def _early_stopping(self) -> bool:
        if self.epsilon >= 0:
            l2_gradient = np.linalg.norm(self.gradients[-1])
            diff = np.abs(self._last_gradient - l2_gradient)
            self._last_gradient = l2_gradient
            if diff < self.epsilon:
                self._current_patience += 1
                return self._current_patience >= self.patience
            else:
                self._current_patience = 0
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

    def copy(self) -> 'NeuralNetwork':
        # return deepcopy(self)
        nn = NeuralNetwork(
            architecture=self.architecture,
            error_calculator=self.error_calculator,
            learning_algorithm=self.learning_algorithm,
            epochs_limit=self.epochs_limit,
            epsilon=self.epsilon,
            patience=self.patience,
            n_init=self.n_init,
        )
        nn._current_network = deepcopy(self._current_network)
        nn._internal_networks = self._internal_networks
        nn._errors = self._errors

        nn.training_curve = self.training_curve
        nn.validation_curve = self.validation_curve
        nn.testing_curve = self.testing_curve
        return nn
