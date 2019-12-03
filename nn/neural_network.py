from __future__ import annotations
from typing import Sequence, Callable, Tuple, List, Optional, Any
import numpy as np

from nn.activation_function import ActivationFunction, sigmoidal
from nn.neuron_layer import NeuronLayer
from nn.number import number
from nn.architecture import Architecture

from enum import Enum


def back_propagation(
    d: Sequence[number],
    nn: NeuralNetwork,
):
    d = np.array(d)
    output_layer_w = np.array(nn.output_layer.w)
    output_layer_out = np.array(nn.output_layer.out)

    delta_k = (d - output_layer_out) * nn.output_layer.fprime
    delta_j = (output_layer_w.T[1:] @ delta_k) * nn.hidden_layer.fprime

    hidden_layer_out = np.array((1,) + tuple(nn.hidden_layer.out))[np.newaxis]
    input__layer_out = np.array((1,) + tuple(nn.input_layer))[np.newaxis]

    nn.output_layer.w += nn.eta * (delta_k * hidden_layer_out.T).T
    nn.hidden_layer.w += nn.eta * (delta_j * input__layer_out.T).T


class ErrorTypes(Enum):
    MEE = 1
    MAE = 2
    MSE = 3
    MIS = 4


class ErrorComputation:

    def __init__(self, identifier):
        self.identifier = identifier

    def __call__(
            self,
            d: Sequence[number],
            out: Sequence[number]
    ):
        if self.identifier == ErrorTypes.MSE:
            return self.mean_square_error(d, out)
        elif self.identifier == ErrorTypes.MEE:
            return self.mean_euclidean_error(d, out)
        elif self.identifier == ErrorTypes.MAE:
            return self.mean_absolute_error(d, out)
        elif self.identifier == ErrorTypes.MIS:
            return self.mismatch_error(d, out)

    @staticmethod
    def mean_square_error(d: Sequence[number], out: Sequence[number]):
        return np.true_divide(np.sum(np.square(np.subtract(d, out))), len(d))

    @staticmethod
    def mean_euclidean_error(d: Sequence[number], out: Sequence[number]):
        return np.true_divide(np.sqrt(np.sum(np.square(np.subtract(d, out)))), len(d))

    @staticmethod
    def mean_absolute_error(d, out):
        return np.true_divide(np.sum(np.abs(np.subtract(d, out))), len(d))

    @staticmethod
    def mismatch_error(d, out):
        return 0 if np.equal(d, np.round(out)).all() else 1



class NeuralNetwork:
    def __init__(
        self,
        architecture: Architecture,
        activation_output: ActivationFunction,
        activation_hidden: ActivationFunction = sigmoidal,
        eta: number = 0.5,
        epoches: int = 1,
        epsilon: number = 0,
        penalty: number = 0,
        error_type: ErrorTypes = ErrorTypes.MSE,
        verbose: int = 0,
        learning_algorithm: Callable = back_propagation,
    ):
        self.activation_hidden: ActivationFunction = activation_hidden
        self.activation_output: ActivationFunction = activation_output
        self.architecture: Architecture = architecture

        self.eta: number = eta
        self.epoches: int = epoches
        self.epsilon: number = epsilon
        self.penalty: number = penalty
        self.error_type: ErrorTypes = error_type
        self.learning_algorithm: Callable = learning_algorithm
        self.verbose: int = verbose

        self.out: Sequence[number] = ()

        self.training_errors: List[number] = []
        self.testing_errors: List[number] = []

        self.hidden_layer: NeuronLayer = NeuronLayer((), self.activation_hidden)
        self.output_layer: NeuronLayer = NeuronLayer((), self.activation_output)
        self.input_layer: Sequence[number] = ()

        self.init()

    def set(self, **kwargs) -> NeuralNetwork:
        self.__dict__.update(**kwargs)
        return self

    def init(self, **kwargs) -> NeuralNetwork:
        self.set(**kwargs)

        self.out = []
        self.hidden_layer = NeuronLayer(
                activation=self.activation_hidden,
                weights=self.architecture.hidden_weights)
        self.output_layer = NeuronLayer(
                activation=self.activation_output,
                weights=self.architecture.output_weights)
        self.input_layer: Sequence[number] = np.zeros(self.architecture.size_input_nodes)
        self.training_errors = []
        self.testing_errors = []

        return self

    def feed_forward(self, *args: number) -> Sequence[number]:
        self.input_layer = args
        self.out = self.output_layer(*self.hidden_layer(*self.input_layer))
        return self.out

    # feed-forward or init
    def __call__(self, *args: number, **kwargs: Any) -> NeuralNetwork:
        if len(args):
            self.feed_forward(*args)
        else:
            self.init(**kwargs)
        return self

    def compute_error(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        error_type: Optional[ErrorTypes] = None
    ) -> number:
        if error_type is None:
            error = self.error_type

        error = 0
        ec: ErrorComputation = ErrorComputation(self.error_type)
        for x, d in patterns:
            self(*x)
            error += ec(d, self.out)
        return error

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

    def train(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        test_patterns: Sequence[Tuple[Sequence[number], Sequence[number]]] = [],
        **kwargs: Any
    ) -> None:
        self.set(**kwargs)

        for _ in range(self.epoches):
            for x, d in patterns:
                self(*x)
                self.learning_algorithm(d, self)

            # needed for constructing the learning curve relative to the testing errors
            self.training_errors.append(self.compute_error(patterns))
            self.testing_errors.append(self.compute_error(test_patterns))

            if self.verbose > 0:
                print(self.training_errors[-1])

            if self.training_errors[-1] <= self.epsilon:
                break

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

    # To obtain the learning curve of both the training and testing set we need to create a function f such that:
    # f(training_set, testing_set) =
    #       for all epoch in epochs
    #           add training error on list of training errors computed on the train set
    #           add testing error on list of testing errors computed on the test set
    # The function needs to take into account also the early stopping hyper-parameter.
    """
    def fill_error_lists(
        self,
        train_set: Sequence[Tuple[Sequence[number], Sequence[number]]],
        test_set: Sequence[Tuple[Sequence[number], Sequence[number]]],
        eta: number,
        epoch_number: int = 0
    ) -> None:
        epoch_number = len(self.training_errors) if epoch_number == 0 else \
            (len(self.training_errors) if 0 < self.epoches < epoch_number else epoch_number)
        for ep in range(epoch_number):
            self.train(train_set, test_set, eta=eta)
            self.test(test_set)
    """
# PROVA PROVA SA SA
