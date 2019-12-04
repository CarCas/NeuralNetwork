from __future__ import annotations
from typing import Sequence, MutableSequence, Tuple, Dict, Optional, Any
import numpy as np
from collections import defaultdict

from nn.activation_function import ActivationFunction, sigmoidal
from nn.neuron_layer import NeuronLayer
from nn.number import number
from nn.architecture import Architecture
from nn.learning_algorithms import LeariningAlgorthm, Online, Batch

from enum import Enum


class ErrorTypes(Enum):
    MEE = 1
    MAE = 2
    MSE = 3
    MIS = 4
    ACC = 5


class ErrorComputation:
    def __init__(self, identifier: ErrorTypes):
        self.identifier: ErrorTypes = identifier

    def __call__(self, d: Sequence[number], out: Sequence[number]) -> Sequence[number]:
        if self.identifier == ErrorTypes.MSE:
            return np.array(self.mean_square_error(d, out))
        elif self.identifier == ErrorTypes.MEE:
            return np.array(self.mean_euclidean_error(d, out))
        elif self.identifier == ErrorTypes.MAE:
            return np.array(self.mean_absolute_error(d, out))
        elif self.identifier == ErrorTypes.MIS:
            return np.array(self.mismatch_error(d, out))
        elif self.identifier == ErrorTypes.ACC:
            return self.accuracy(d, out)
        return np.array(-1)

    def post(self, error: Sequence[number], len: int) -> Sequence[number]:
        if self.identifier == ErrorTypes.MEE:
            error = np.sqrt(error)
        return np.true_divide(error, len)

    @staticmethod
    def mean_square_error(d: Sequence[number], out: Sequence[number]) -> Sequence[number]:
        return np.square(np.subtract(d, out))

    @staticmethod
    def mean_euclidean_error(d: Sequence[number], out: Sequence[number]) -> Sequence[number]:
        return np.square(np.subtract(d, out))

    @staticmethod
    def mean_absolute_error(d: Sequence[number], out: Sequence[number]) -> Sequence[number]:
        return np.abs(np.subtract(d, out))

    @staticmethod
    def mismatch_error(d: Sequence[number], out: Sequence[number]) -> Sequence[number]:
        return np.not_equal(d, np.round(out)).astype(float)

    @staticmethod
    def accuracy(d: Sequence[number], out: Sequence[number]) -> Sequence[number]:
        return np.equal(d, np.round(out)).astype(float)


class NeuralNetwork:
    '''
    Params and istance variable
    ----------------------------------------
    architecture: sizes of the network
    eta: learning rate
    epochs_limit: max possible number of epoch during a train call
    activation_output: activation function of output layer
    activation_hidden: activation function of hidden layers
    learning_algorithm: batch or online (from nn import Online, Batch)
    epsilon: early stopping condition, after a train if current error_types[0] < epsilon -> stop
    epsilon_relative: ealy stopping condition, if error decrease too slow stop (wrt mean of last 10 measures)
    penality: TODO
    error_types: Sequence of error types of learning to curve to save during training

    Just params
    ----------------------------------------
    out: last output of the nn (out == output_layer.out)
    input: the last input of the nn
    training_errors: dictionary, each key ErrorTypes refer to the respective learning curve for given patterns in train
    testing_errors: dictionary, each key ErrorTypes refer to the respective learning curve for given test_patterns in train

    hidden_layer: hidden_layer with last output and weights of the nn
    output_layer: output_layer with last output and weights of the nn


    Debug params
    ----------------------------------------
    verbose: to print some information during training
    seed: to set a random seed


    Public methods
    ----------------------------------------
    __call__: execute feed forward
              or if no *args is specified: reset the network with te possibility to change parameters with **kwargs
    train: train the network
    compute_error: compute the error for the patterns given and the specified error_type (defaut is self.error_types[0])

    '''
    def __init__(
        self,
        architecture: Architecture,
        activation_output: ActivationFunction,
        activation_hidden: ActivationFunction = sigmoidal,
        learning_algorithm: LeariningAlgorthm = Online(),
        eta: number = 0.5,
        epochs_limit: int = 1,
        epsilon: number = 0,
        epsilon_relative: number = -1,
        penalty: number = 0,
        error_types: Sequence[ErrorTypes] = (ErrorTypes.MSE,),
        seed: Optional[int] = None,
        verbose: int = 0,
    ):
        if seed:
            np.random.seed(seed)

        self.activation_hidden: ActivationFunction = activation_hidden
        self.activation_output: ActivationFunction = activation_output
        self.architecture: Architecture = architecture

        self.eta: number = eta
        self.epochs_limit: int = epochs_limit
        self.epsilon: number = epsilon
        self.epsilon_relative: number = epsilon_relative
        self.penalty: number = penalty
        self.error_types: MutableSequence[ErrorTypes] = list(error_types)
        self.verbose: int = verbose
        self.learning_algorithm: LeariningAlgorthm = learning_algorithm

        self.out: Sequence[number] = ()

        self.training_errors: defaultdict[ErrorTypes, MutableSequence[Sequence[number]]] = defaultdict(lambda: [])
        self.testing_errors: defaultdict[ErrorTypes, MutableSequence[Sequence[number]]] = defaultdict(lambda: [])

        self.hidden_layer: NeuronLayer = NeuronLayer((), self.activation_hidden)
        self.output_layer: NeuronLayer = NeuronLayer((), self.activation_output)
        self.input: Sequence[number] = ()

        self._init()

    def set(self, **kwargs) -> NeuralNetwork:
        self.__dict__.update(**kwargs)
        return self

    # feed-forward or _init
    def __call__(self, *args: number, **kwargs: Any) -> NeuralNetwork:
        if len(args):
            self.feed_forward(*args)
        else:
            self._init(**kwargs)
        return self

    def train(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        test_patterns: Sequence[Tuple[Sequence[number], Sequence[number]]] = (),
        **kwargs: Any
    ) -> None:
        self.set(**kwargs)

        for _ in range(self.epochs_limit):
            self.learning_algorithm(patterns, self)

            self._append_learning_curve_errors(patterns, test_patterns)

            if self._early_stopping():
                break

    def compute_error(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        error_type: Optional[ErrorTypes] = None
    ) -> Sequence[number]:
        if error_type is None:
            error_type = self.error_types[0]

        error = np.array(0)
        ec: ErrorComputation = ErrorComputation(error_type)
        for x, d in patterns:
            self(*x)
            error = np.add(error, ec(d, self.out))
        return ec.post(error, len(patterns))

    # TODO applicare regularization al learning
    def compute_error_with_regularization(
        self,
        starting_error
    ) -> number:
        penalty_norm: number = 0
        for weights in self.output_layer.w:
            penalty_norm += np.linalg.norm(weights)
        for weights in self.hidden_layer.w:
            penalty_norm += np.linalg.norm(weights)
        error = starting_error + np.square(penalty_norm) * self.penalty
        return error

    def _init(self, **kwargs) -> NeuralNetwork:
        self.set(**kwargs)

        self.out = []
        self.hidden_layer = NeuronLayer(
                activation=self.activation_hidden,
                weights=self.architecture.hidden_weights)
        self.output_layer = NeuronLayer(
                activation=self.activation_output,
                weights=self.architecture.output_weights)
        self.input: Sequence[number] = np.zeros(self.architecture.size_input_nodes)
        self.training_errors = defaultdict(lambda: [])
        self.testing_errors = defaultdict(lambda: [])

        return self

    def feed_forward(self, *args: number) -> Sequence[number]:
        self.input = args
        self.out = self.output_layer(*self.hidden_layer(*self.input))
        return self.out

    def _early_stopping(self) -> bool:
        if np.less_equal(self.training_errors[self.error_types[0]][-1], self.epsilon).all():
            self._log('early stop', 'error < epsilon')
            return True

        if len(self.training_errors) > 10:
            present = self.training_errors[self.error_types[0]][-1]
            past = np.mean(self.training_errors[self.error_types[0]][-10:-2], axis=0)
            variation = np.abs(np.subtract(1, np.divide(present, past)))

            if np.less_equal(variation, self.epsilon_relative).all():
                self._log('early stop', 'error decreasing too slow')
                return True

        return False

    def _append_learning_curve_errors(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        test_patterns: Sequence[Tuple[Sequence[number], Sequence[number]]] = (),
    ) -> None:
        # needed for constructing the learning curve relative to the testing errors
        for error_type in self.error_types:
            self.training_errors[error_type].append(self.compute_error(patterns, error_type))
            self.testing_errors[error_type].append(self.compute_error(test_patterns, error_type))

        self._log(self.training_errors[self.error_types[0]][-1])

    def _log(self, *args: Any) -> None:
        if self.verbose > 0:
            print(*args)

    def get_training_errors(self):
        return self.training_errors

    def get_testing_errors(self):
        return self.testing_errors

    def __len__(self):
        return len(self.out)

    def __getitem__(self, index: int):
        return self.out[index]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self):
            result = self[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
