from __future__ import annotations
from typing import Sequence, MutableSequence, Tuple, List, Optional, Any
import numpy as np
from collections import defaultdict
from copy import deepcopy

from nn.types import Pattern, Architecture, BaseNeuralNetwork, ActivationFunction, LearningAlgorithm
from nn.error_types import ErrorTypes, ErrorComputation
from nn.activation_functions import sigmoid


class NeuralNetwork:
    '''
    Params and istance variables
    ----------------------------------------
    activation: activation function of output layer
    architecture: sizes of the network
    activation_hidden: activation function of hidden layers
    eta: learning rate
    epochs_limit: max possible float of epoch during a train call
    epsilon: early stopping condition, after a train if current error_types[0] < epsilon -> stop
    epsilon_relative: ealy stopping condition, if error decrease too slow stop (wrt mean of last 10 measures)
    penality: TODO
    error_types: Sequence of error types of learning to curve to save during training
    n_init: float to different initialization of the network during training, the best one is choose

    Istance variables
    ----------------------------------------
    out: last output of the nn (out == output_layer.out)
    input: the last input of the nn
    training_errors: dictionary, each key ErrorTypes refer to the
                     respective learning curve for given patterns in train
    testing_errors : dictionary, each key ErrorTypes refer to the
                     respective learning curve for given test_patterns in train

    hidden_layer: hidden_layer with last output and weights of the nn
    output_layer: output_layer with last output and weights of the nn


    Debug params
    ----------------------------------------
    seed: to set a random seed
    verbose: to print some information during training


    Public methods
    ----------------------------------------
    __call__: execute feed forward
    set: reset the network, if some parameters are passed their are setted in the network
    train: train the network, fill `training_errors` and `testing_errors` instance variable
    compute_error: compute the error for the patterns given and the specified
                   error_type (defaut is self.error_types[0])

    Public properties
    ----------------------------------------
    current_training_error: return the last computed error on training
                                   for the first specified error (error_types[0])
    '''
    def __init__(
        self,
        activation: ActivationFunction,
        architecture: Architecture,
        activation_hidden: ActivationFunction = sigmoid,
        eta: float = 0.5,
        epochs_limit: int = 1,
        epsilon: float = -1,
        epsilon_relative: float = -1,
        penalty: float = 0,
        error_types: Sequence[ErrorTypes] = (ErrorTypes.MSE,),
        n_init: int = 1,
        learning_algorithm: LearningAlgorithm = LearningAlgorithm.BATCH,

        seed: Optional[int] = None,
        verbose: int = 0,
        **kwargs
    ) -> None:
        if seed is not None:
            np.random.seed(seed=seed)

        self.architecture = architecture
        self._nn: BaseNeuralNetwork = architecture(
            eta=eta,
            activation=activation,
            activation_hidden=activation_hidden
        )

        self.activation: ActivationFunction = activation
        self.activation_hidden: ActivationFunction = activation_hidden

        self.eta: float = eta
        self.epochs_limit: int = epochs_limit
        self.epsilon: float = epsilon
        self.epsilon_relative: float = epsilon_relative
        self.penalty: float = penalty
        self.error_types: MutableSequence[ErrorTypes] = list(error_types)
        self.verbose: int = verbose
        self.n_init: int = n_init
        self.learning_algorithm: LearningAlgorithm = learning_algorithm

        self.out: Sequence[Sequence[float]] = []

        self.training_errors: defaultdict[ErrorTypes, MutableSequence[float]] = defaultdict(lambda: [])
        self.testing_errors: defaultdict[ErrorTypes, MutableSequence[float]] = defaultdict(lambda: [])

    def set(self, **kwargs) -> NeuralNetwork:
        self.__dict__.update(**kwargs)
        self.__init__(**self.__dict__)
        return self

    # feed-forward
    def __call__(self, *args: Sequence[float]) -> Sequence[Sequence[float]]:
        return self._nn(*args)

    def train(
        self,
        patterns: Sequence[Pattern],
        test_patterns: Sequence[Pattern] = (),
    ) -> None:
        container_choosen_nn: List[NeuralNetwork] = []

        for i_init in range(self.n_init):
            for i_epoch in range(self.epochs_limit):
                if self.learning_algorithm == LearningAlgorithm.BATCH:
                    self._nn.train(patterns)

                elif self.learning_algorithm == LearningAlgorithm.ONLINE:
                    for pattern in patterns:
                        self._nn.train([pattern])

                self._append_learning_curve_errors(patterns, test_patterns)

                self._log(2, 'train', [('init', i_init),
                                       ('epoch', i_epoch),
                                       ('error', self.current_training_error)])

                if self._early_stopping():
                    break

            self._log(1, 'train', [('init', i_init),
                                   ('error', self.current_training_error)])

            self._choose_nn(container_choosen_nn)
            self.set()

        self._copy(container_choosen_nn)

    def compute_error(
        self,
        patterns: Sequence[Pattern],
        error_type: Optional[ErrorTypes] = None
    ) -> float:
        if not len(patterns):
            return np.nan

        if error_type is None:
            error_type = self.error_types[0]

        error = np.array(0)
        ec: ErrorComputation = ErrorComputation(error_type)
        for x, d in patterns:
            self(x)
            error = np.add(error, ec(d, self._nn.out[-1]))
        return ec.post(error, len(patterns))

    # # TODO applicare regularization al learning
    # def compute_error_with_regularization(
    #     self,
    #     starting_error
    # ) -> float:
    #     penalty_norm: float = 0
    #     for weights in self.output_layer.w:
    #         penalty_norm += np.linalg.norm(weights)
    #     for weights in self.hidden_layer.w:
    #         penalty_norm += np.linalg.norm(weights)
    #     error = starting_error + np.square(penalty_norm) * self.penalty
    #     return error

    @property
    def current_training_error(self) -> float:
        return self.training_errors[self.error_types[0]][-1]

    def _early_stopping(self) -> bool:
        if self.epsilon != -1 and np.less_equal(self.current_training_error, self.epsilon).all():
            self._log(1, 'early_stopping', [('early stop', 'error < epsilon')])
            return True

        if self.epsilon_relative != -1 and len(self.training_errors) > 10:
            present = self.current_training_error
            past = np.mean(self.training_errors[self.error_types[0]][-10:-2], axis=0)
            variation = np.mean(np.abs(np.subtract(1, np.true_divide(present, past))))

            if variation < self.epsilon_relative:
                self._log(1, 'early_stopping', [('early stop', 'error decreasing too slow')])
                return True

        return False

    def _append_learning_curve_errors(
        self,
        patterns: Sequence[Pattern],
        test_patterns: Sequence[Pattern] = (),
    ) -> None:
        # needed for constructing the learning curve relative to the testing errors
        for error_type in self.error_types:
            self.training_errors[error_type].append(self.compute_error(patterns, error_type))
            self.testing_errors[error_type].append(self.compute_error(test_patterns, error_type))

    def _choose_nn(self, nn_container: List[NeuralNetwork]) -> None:
        if not len(nn_container):
            nn_container.append(deepcopy(self))

        elif np.mean(self.current_training_error) < np.mean(nn_container[0].current_training_error):
            nn_container[0] = deepcopy(self)

    def _copy(self, nn_container: List[NeuralNetwork]) -> None:
        if len(nn_container):
            self.__dict__.update(**(nn_container[0].__dict__))

    def _log(
        self,
        verbose: int = 10,
        method_name: Optional[str] = None,
        info: Sequence[Tuple[str, Any]] = (),
    ) -> None:
        if verbose <= self.verbose:
            result: MutableSequence[str] = [str(verbose)]
            if method_name is not None:
                result.append(method_name)
            for key, value in info:
                result.append(key + ': ' + str(value))
            print(' - '.join(result))

    def get_training_errors(self):
        return self.training_errors

    def get_testing_errors(self):
        return self.testing_errors
