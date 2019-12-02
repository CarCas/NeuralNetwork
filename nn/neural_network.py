from __future__ import annotations
from typing import Sequence, Callable, Tuple, List
import numpy as np

from nn.activation_function import ActivationFunction
from nn.neuron_layer import NeuronLayer
from nn.number import number
from nn.architecture import Architecture

from enum import Enum


def back_propagation(
    d: Sequence[number],
    eta: number,
    nn: NeuralNetwork,
):
    delta_k = np.subtract(d, nn.output_layer.out) * nn.output_layer.fprime
    delta_j = np.array(nn.output_layer.w).T[1:] @ delta_k * nn.hidden_layer.fprime

    nn.output_layer.w += eta * (delta_k * np.array((1,) + tuple(nn.hidden_layer.out))[np.newaxis].T).T
    nn.hidden_layer.w += eta * (delta_j * np.array((1,) + tuple(nn.input_layer))[np.newaxis].T).T


class ErrorTypes(Enum):
    MEE = 1
    MAE = 2
    MSE = 3


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

    @staticmethod
    def mean_square_error(d: Sequence[number], out: Sequence[number]):
        return np.true_divide(np.sum(np.square(np.subtract(d, out))), len(d))

    @staticmethod
    def mean_euclidean_error(d: Sequence[number], out: Sequence[number]):
        return np.true_divide(np.sqrt(np.sum(np.square(np.subtract(d, out)))), len(d))

    @staticmethod
    def mean_absolute_error(d, out):
        return np.true_divide(np.sum(np.abs(np.subtract(d, out))), len(d))


class NeuralNetwork:
    def __init__(
        self,
        activation: ActivationFunction,
        architecture: Architecture,
        early_stopping: int = 0,
        epsilon: number = 0,
        penalty: number = 0.01,
        learning_algorithm: Callable = back_propagation
    ):

        self.activation = activation
        self.learning_algorithm = learning_algorithm

        self.hidden_layer = NeuronLayer(
                activation=activation,
                weights=architecture.hidden_weights)
        self.output_layer = NeuronLayer(
                activation=activation,
                weights=architecture.output_weights)

        self.input_layer: Sequence[number] = np.zeros(architecture.size_input_nodes)
        self.out: Sequence[number]

        self.training_errors: List[number] = []
        self.testing_errors: List[number] = []
        self.loss_errors: List[number] = []

        self.error: ErrorTypes = ErrorTypes.MSE
        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.penalty = penalty

    # feed-forward
    def __call__(
        self,
        *args: number
    ) -> Sequence[number]:
        self.input_layer = args
        self.out = self.output_layer(*self.hidden_layer(*self.input_layer))
        return self.out

    def compute_error(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]]
    ) -> number:
        error = 0
        ec: ErrorComputation = ErrorComputation(self.error)
        for x, d in patterns:
            self(*x)
            error += ec(d, self.out)
        return error

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

    def _train_on_patterns(
        self,
        in_patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        test_patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        in_eta: number = 0.5,
    ) -> (number, number):
        for x, d in in_patterns:
            self(*x)
            self.learning_algorithm(d, in_eta, self)
        error = self.compute_error(in_patterns)
        if self.penalty != 0:
            loss = self.compute_error_with_regularization(error)
            self.loss_errors.append(loss)
            self.training_errors.append(error)
            # needed for constructing the learning curve relative to the testing errors
            self.testing_errors.append(self.test(test_patterns))
            return error, loss
        else:
            self.training_errors.append(error)
            # needed for constructing the learning curve relative to the testing errors
            self.testing_errors.append(self.test(test_patterns))
            return error, None

    def train(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        test_patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        eta: number = 0.5,
    ) -> None:
        if self.early_stopping > 0 and self.epsilon > 0:
            for _ in range(self.early_stopping):
                if self._train_on_patterns(patterns, test_patterns, eta)[1] < self.epsilon:
                    break
        elif self.early_stopping > 0:
            for _ in range(self.early_stopping):
                self._train_on_patterns(patterns, test_patterns, eta)
        elif self.epsilon > 0:
            while self._train_on_patterns(patterns, test_patterns, eta)[1] >= self.epsilon:
                pass
        else:
            self._train_on_patterns(patterns, test_patterns, eta)

    def test(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]]
    ) -> number:
        error = self.compute_error(patterns)
        self.testing_errors.append(error)
        return error

    def get_training_errors(self):
        return self.training_errors

    def get_testing_errors(self):
        return self.testing_errors

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
            (len(self.training_errors) if 0 < self.early_stopping < epoch_number else epoch_number)
        for ep in range(epoch_number):
            self.train(train_set, test_set, eta=eta)
            self.test(test_set)
    """

# PROVA PROVA SA SA
