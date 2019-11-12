from __future__ import annotations
from typing import Sequence, Optional, Callable, Tuple, List
import random

from nn.activation_function import ActivationFunction
from nn.neuron_layer import NeuronLayer

from enum import Enum


def back_propagation(
    d: Sequence[float],
    eta: float,
    nn: NeuralNetwork,
):
    delta_k = []
    for k in range(len(nn.output_layer)):
        n_k = nn.output_layer[k]
        delta_k.append((d[k] - n_k.out) * n_k.fprime)

    delta_j = []
    for j in range(len(nn.hidden_layer)):
        n_j = nn.hidden_layer[j]
        wk_j = nn.output_layer.w_from(j)
        delta = 0
        for i in range(len(delta_k)):
            delta += delta_k[i] * wk_j[i]
        delta_j.append(delta * n_j.fprime)

    for k in range(len(nn.output_layer)):
        n_k = nn.output_layer[k]
        delta_w = []
        for i in range(len(nn.hidden_layer)):
            o_i = nn.hidden_layer[i].out
            delta_w.append(delta_k[k] * o_i)
        for i in range(len(delta_w)):
            n_k.w[i] += eta * delta_w[i]

    for j in range(len(nn.hidden_layer)):
        n_j = nn.hidden_layer[j]
        delta_w = []
        for i in range(len(nn.input)):
            o_i = nn.input[i]
            delta_w.append(delta_j[j] * o_i)
        for i in range(len(delta_w)):
            n_j.w[i] += eta * delta_w[i]


class ErrorTypes(Enum):
    MEE = 1
    MAE = 2
    MSE = 3


class ErrorComputation:

    def __init__(self, identifier):
        self.identifier = identifier

    def __call__(
            self,
            d: Sequence[float],
            out: Sequence[float]
    ):
        if self.identifier == ErrorTypes.MSE:
            return self.mean_square_error(d, out)
        elif self.identifier == ErrorTypes.MEE:
            return self.mean_euclidean_error(d, out)
        elif self.identifier == ErrorTypes.MAE:
            return self.mean_absolute_error(d, out)

    @staticmethod
    def mean_square_error(d, out):
        error = 0
        for i in range(len(d)):
            error += 0.5 * (d[i] - out[i]) ** 2
        return error

    @staticmethod
    def mean_euclidean_error(d, out):
        error = 0
        # todo to implement
        return error

    @staticmethod
    def mean_absolute_error(d, out):
        error = 0
        # todo to implement
        return error


class NeuralNetwork:
    def __init__(
        self,
        activation: ActivationFunction,
        architecture: Architecture,
        early_stopping: int = 0,
        epsilon: float = 0,
        learning_algorithm: Callable = back_propagation
    ):

        self.activation = activation
        self.learning_algorithm = learning_algorithm

        self.hidden_layer = NeuronLayer(
                size=architecture.number_hidden,
                activation=activation,
                weights=architecture.hidden_weights,
                bias=architecture.hidden_bias)
        self.output_layer = NeuronLayer(
                size=architecture.number_outputs,
                activation=activation,
                weights=architecture.output_weights,
                bias=architecture.output_bias)

        self.input: Sequence[float]
        self.out: Sequence[float]

        self.training_errors: List[float] = []
        self.testing_errors: List[float] = []

        self.error: ErrorTypes = ErrorTypes.MSE
        self.early_stopping = early_stopping
        self.epsilon = epsilon

    # feed-forward william carmine
    def __call__(
        self,
        *args: float
    ) -> Sequence[float]:
        self.input = tuple(args)
        self.out = self.output_layer(*self.hidden_layer(*self.input))
        return self.out

    def compute_error(
        self,
        patterns: Sequence[Tuple[Sequence[float], Sequence[float]]]
    ) -> float:
        error = 0
        ec: ErrorComputation = ErrorComputation(self.error)
        for x, d in patterns:
            self(*x)
            error += ec(d, self.out)
        return error

    def train(
        self,
        patterns: Sequence[Tuple[Sequence[float], Sequence[float]]],
        test_patterns: Sequence[Tuple[Sequence[float], Sequence[float]]],
        eta: float = 0.5
    ) -> None:
        def train_on_patterns(in_patterns, in_eta):
            for x, d in in_patterns:
                self(*x)
                self.learning_algorithm(d, in_eta, self)
            in_error = self.compute_error(in_patterns)
            self.training_errors.append(in_error)
            # needed for constructing the learning curve relative to the testing errors
            self.testing_errors.append(self.test(test_patterns))
            return in_error

        if self.early_stopping > 0 and self.epsilon > 0:
            for _ in range(self.early_stopping):
                if train_on_patterns(patterns, eta) < self.epsilon:
                    break
        elif self.early_stopping > 0:
            for _ in range(self.early_stopping):
                train_on_patterns(patterns, eta)
        elif self.epsilon > 0:
            while train_on_patterns(patterns, eta) >= self.epsilon:
                pass

    def test(
        self,
        patterns: Sequence[Tuple[Sequence[float], Sequence[float]]]
    ) -> float:
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
    def fill_error_lists(
        self,
        train_set: Sequence[Tuple[Sequence[float], Sequence[float]]],
        test_set: Sequence[Tuple[Sequence[float], Sequence[float]]],
        eta: float,
        epoch_number: int = 0
    ) -> None:
        epoch_number = len(self.training_errors) if epoch_number == 0 else \
            (len(self.training_errors) if 0 < self.early_stopping < epoch_number else epoch_number)
        for ep in range(epoch_number):
            self.train(train_set, test_set, eta=eta)
            self.test(test_set)

    class Architecture:
        def __init__(
            self,
            number_inputs: int,
            number_hidden: int,
            number_outputs: int,

            hidden_weights: Optional[Sequence[List[float]]] = None,
            output_weights: Optional[Sequence[List[float]]] = None,

            hidden_bias: Optional[float] = None,
            output_bias: Optional[float] = None,

            threshold: Optional[int] = None,
            range_weights: Optional[float] = None,
        ):

            self.number_inputs: int = number_inputs
            self.number_hidden: int = number_hidden
            self.number_outputs: int = number_outputs

            self.threshold: int = threshold
            self.range_weights: float = range_weights

            self.hidden_bias: float
            self.output_bias: float

            self.hidden_weights: Sequence[List[float]]
            self.output_weights: Sequence[List[float]]

            if hidden_weights is None:
                _hidden_weights = []
                for _ in range(self.number_hidden):
                    w = []
                    for _ in range(self.number_inputs):
                        # starting values updated:
                        # if #inputs > threshold ==> quite as before: rand * 0.4 - 0.2
                        # else #inputs <= threshold ==> (rand * 0.4 - 0.2) * 2 / #inputs

                        w.append(random.uniform(- self.range_weights, self.range_weights)
                                 if self.number_inputs > self.threshold else
                                 random.uniform(- self.range_weights, self.range_weights) * 2 / self.number_inputs)
                    _hidden_weights.append(w)
                self.hidden_weights = _hidden_weights
            else:
                self.hidden_weights = hidden_weights

            if output_weights is None:
                _output_weights = []
                for _ in range(self.number_outputs):
                    w = []
                    for _ in range(self.number_hidden):
                        w.append(random.uniform(- self.range_weights, self.range_weights)
                                 if self.number_hidden > self.threshold else
                                 random.uniform(- self.range_weights, self.range_weights) * 2 / self.number_hidden)
                    _output_weights.append(w)
                self.output_weights = _output_weights
            else:
                self.output_weights = output_weights

            if hidden_bias is None:
                self.hidden_bias = random.uniform(- self.range_weights, self.range_weights) \
                    if self.number_inputs > self.threshold else \
                    random.uniform(- self.range_weights, self.range_weights) * 2 / self.number_inputs
            else:
                self.hidden_bias = hidden_bias

            if output_bias is None:
                self.output_bias = random.uniform(- self.range_weights, self.range_weights) \
                     if self.number_hidden > self.threshold else \
                     random.uniform(- self.range_weights, self.range_weights) * 2 / self.number_hidden
            else:
                self.output_bias = output_bias

