from __future__ import annotations
from typing import Sequence, Callable, MutableSequence, Tuple, List, Optional, Any
import numpy as np

from nn.activation_function import ActivationFunction, sigmoidal
from nn.neuron_layer import NeuronLayer
from nn.number import number
from nn.architecture import Architecture
from nn.learning_algorithms import LeariningAlgorthm, Online

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
    def __init__(
        self,
        architecture: Architecture,
        activation_output: ActivationFunction,
        activation_hidden: ActivationFunction = sigmoidal,
        eta: number = 0.5,
        epochs_limit: int = 1,
        epsilon: number = 0,
        epsilon_relative: number = -1,
        penalty: number = 0,
        error_type: ErrorTypes = ErrorTypes.MSE,
        verbose: int = 0,
        learning_algorithm: LeariningAlgorthm = Online(),
        seed: Optional[int] = None
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
        self.error_type: ErrorTypes = error_type
        self.verbose: int = verbose
        self.learning_algorithm: LeariningAlgorthm = learning_algorithm

        self.out: Sequence[number] = ()

        self.training_errors: MutableSequence[Sequence[number]] = []
        self.testing_errors: MutableSequence[Sequence[number]] = []

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
    ) -> Sequence[number]:
        if error_type is None:
            error_type = self.error_type

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

    def train(
        self,
        patterns: Sequence[Tuple[Sequence[number], Sequence[number]]],
        test_patterns: Sequence[Tuple[Sequence[number], Sequence[number]]] = (),
        **kwargs: Any
    ) -> None:
        self.set(**kwargs)

        for _ in range(self.epochs_limit):
            self.learning_algorithm(patterns, self)

            # needed for constructing the learning curve relative to the testing errors
            self.training_errors.append(self.compute_error(patterns))
            self.testing_errors.append(self.compute_error(test_patterns))

            if self.verbose > 0:
                print(self.training_errors[-1])

            if self._check_epsilon():
                break

    def _check_epsilon(self) -> bool:
        if np.less_equal(self.training_errors[-1], self.epsilon).all():
            return True

        if len(self.training_errors) > 10:
            present = self.training_errors[-1]
            past = np.mean(self.training_errors[-10:-2], axis=0)
            variation = np.abs(np.subtract(1, np.divide(present, past)))

            return np.less_equal(variation, self.epsilon_relative).all()

        return False

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
