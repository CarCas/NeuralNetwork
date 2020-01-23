import random
from typing import List, Optional, Sequence, Tuple, MutableSequence
import numpy as np
import itertools
from nn.types import Architecture, ActivationFunction, Pattern
from nn.learning_algorithm import LearningAlgorithm
from nn.error_calculator import ErrorCalculator
from nn import NeuralNetwork, MultilayerPerceptron, sigmoid
from nn.playground.utilities import read_monk


def validation(
    nn: NeuralNetwork,
    training_set: Sequence[Pattern],
    validation_set: Sequence[Pattern],
    error_calculator: ErrorCalculator = ErrorCalculator.MSE
) -> Tuple[int, float, float]:
    nn.fit(training_set)

    learning_curve_training = nn.compute_learning_curve(training_set, error_calculator)
    learning_curve_validation = nn.compute_learning_curve(validation_set, error_calculator)

    idx, score = error_calculator.choose(learning_curve_validation)

    return idx, score, learning_curve_training[idx]


def shuffle(patterns: Sequence[Pattern], seed: Optional[int] = None) -> Sequence[Pattern]:
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(patterns)


def _compute_size_given_percentage(dataset: Sequence[Pattern], percentage: float):
    assert(0 < percentage < 1)
    return int(np.round(len(dataset) * percentage))


# return a tuple containing two dataset, the first one contains percentage elements
def split_dataset(
    dataset: Sequence[Pattern],
    percentage: float = 0.7,
    size: Optional[int] = None,
) -> Tuple[Sequence[Pattern], Sequence[Pattern]]:
    if size is not None:
        len_training = size
    else:
        len_training = _compute_size_given_percentage(dataset, percentage)

    return dataset[:len_training], dataset[len_training:]


def k_fold_CV(
    nn: NeuralNetwork,
    dataset: Sequence[Pattern],
    cv: int = 5,
    *,
    error_calculator: ErrorCalculator = ErrorCalculator.MSE,
    to_shuffle: bool = False,
    seed: Optional[int] = None,
) -> Tuple[float, float, Sequence[Tuple[int, float, float]]]:
    if to_shuffle:
        dataset = shuffle(dataset, seed)

    len_training = int(np.round(len(dataset)*(cv-1)/cv))
    shift_size = int(np.round(len_training))

    scores: MutableSequence[Tuple[int, float, float]] = []

    for i in range(cv):
        training_set, validation_set = split_dataset(np.roll(dataset, shift_size*i), size=len_training)
        scores.append(validation(nn.set(), training_set, validation_set, error_calculator=error_calculator))

    scores_1 = list(map(lambda x: x[1], scores))

    score = float(np.mean(scores_1))
    std = float(np.std(scores_1))

    return score, std, scores


# class Validation:

#     def __init__(self,
#                  dataset: Sequence[Pattern]):
#         self.architecture: List[Architecture] = []
#         self.dataset = random.sample(dataset, len(dataset))
#         self.training_errors: List[Sequence[float]] = []
#         self.min_training_errors: List[float] = []
#         self.avg_training_errors: List[float] = []
#         self.testing_errors: List[Sequence[float]] = []
#         self.min_testing_errors: List[float] = []
#         self.avg_testing_errors: List[float] = []
#         self.argmin_testing_errors: List[int] = []

#         self.averages: List[float] = []
#         self.minimums: List[float] = []

#     def update_error_lists(self, start: int, end: int, nn: NeuralNetwork):
#         ts_set = self.dataset[start:end]
#         tr_set = list(self.dataset[:start])
#         if end != len(self.dataset):
#             tr_set.extend(itertools.chain(self.dataset[end:]))
#         train_errs = nn.compute_learning_curve(tr_set)
#         test_errs = nn.compute_learning_curve(ts_set)
#         self.training_errors.append(train_errs)
#         self.avg_training_errors.append(np.mean(train_errs).mean())
#         el = np.argmin(test_errs)
#         assert isinstance(el, np.int64)
#         self.min_training_errors.append(train_errs[el])
#         self.testing_errors.append(test_errs)
#         self.avg_testing_errors.append(np.mean(test_errs).mean())
#         self.min_testing_errors.append(test_errs[el])
#         self.argmin_testing_errors.append(el)

#     def nn_calling(self, architecture: Architecture, learning_algorithm: LearningAlgorithm,
#                    error_calculator: ErrorCalculator, penalty: float,
#                    epochs_limit: int, epsilon: float, seed: Optional[int], dim_mini_batch: int = -1,
#                    kfold: Optional[int] = None):
#         nn = NeuralNetwork(architecture=architecture, error_calculator=error_calculator,
#                            learning_algorithm=learning_algorithm, penalty=penalty,
#                            epochs_limit=epochs_limit, epsilon=epsilon, seed=seed, dim_mini_batch=dim_mini_batch)
#         if kfold is None:
#             self.update_error_lists(0, round(len(self.dataset) * 4 / 5), nn)
#         else:
#             indexes = np.ceil((len(self.dataset) / kfold))
#             for i in range(indexes - 1):
#                 self.update_error_lists(i * kfold, (i + 1) * kfold, nn)
#             self.update_error_lists((indexes - 1) * kfold, len(self.dataset), nn)
#         avg_err = sum(err for (_, err) in self.avg_training_errors) / len(self.avg_training_errors)
#         min_err = min(err for (_, err) in self.min_training_errors)
#         self.averages.append(avg_err)
#         self.minimums.append(min_err)

#     @staticmethod
#     def architecture_create(
#             layer_sizes: Tuple[int],
#             activation: ActivationFunction,
#             activation_hidden: ActivationFunction = sigmoid,
#             eta: float = 0.5,
#             alpha: float = 0,
#             alambd: float = 0.1,

#             layers: Optional[Sequence[Sequence[Sequence[float]]]] = None
#     ) -> Architecture:
#         mp = MultilayerPerceptron(*layer_sizes, activation=activation, activation_hidden=activation_hidden,
#                                   eta=eta, alpha=alpha, alambd=alambd, layers=layers)
#         return mp

#     def list_archs(
#             self,
#             layer_sizes: List[Tuple[int]],
#             activation: List[ActivationFunction],
#             activation_hidden: List[ActivationFunction] = sigmoid,
#             eta: List[float] = 0.5,
#             alpha: List[float] = 0,
#             alambd: List[float] = 0.1,

#             layers: Optional[Sequence[Sequence[Sequence[float]]]] = None
#     ) -> List[Architecture]:
#         archs = [self.architecture_create(layer_sizes=ls, activation=act, activation_hidden=ah,
#                                           eta=e, alpha=a, alambd=lmb, layers=layers)
#                  for ls in layer_sizes
#                  for act in activation
#                  for ah in activation_hidden
#                  for e in eta
#                  for a in alpha
#                  for lmb in alambd]
#         return archs

#     def grid_search(
#             self,
#             layer_sizes: List[Tuple[int]],
#             activation: List[ActivationFunction],
#             activation_hidden: List[ActivationFunction],
#             learning_algorithm: List[LearningAlgorithm],
#             error_calculator: List[ErrorCalculator],
#             penalty: List[float],
#             eta: List[float],
#             epochs_limit: List[int],
#             epsilon: List[float],
#             alpha: List[float],
#             alambd: List[float],
#             dim_mini_batches: List[int],
#             seed: Optional[int],
#             layers: Optional[Sequence[Sequence[Sequence[float]]]] = None):
#         self.architecture.extend(self.list_archs(layer_sizes=layer_sizes, activation=activation,
#                                                  activation_hidden=activation_hidden, eta=eta, alpha=alpha,
#                                                  alambd=alambd, layers=layers))
#         for arch in self.architecture:
#             for la in learning_algorithm:
#                 for ec in error_calculator:
#                     for p in penalty:
#                         for el in epochs_limit:
#                             for eps in epsilon:
#                                 for mb in dim_mini_batches:
#                                     self.nn_calling(arch, la, ec, p, el, eps, seed, mb)

#         return self.training_errors, self.testing_errors, self.averages, self.minimums


# def validate(
#         dataset: Sequence[Pattern],
#         layer_sizes: List[Tuple[int]],
#         activation: List[ActivationFunction],
#         learning_algorithm: List[LearningAlgorithm],
#         error_calculator: List[ErrorCalculator],
#         dim_mini_batch: List[int],
#         epochs_limit: List[int],
#         epsilon: List[float],
#         penalty: List[float],
#         seed: Optional[int],
#         activation_hidden: List[ActivationFunction] = sigmoid,
#         eta: List[float] = 0.5,
#         alpha: List[float] = 0,
#         alambd: List[float] = 0.1,

#         layers: Optional[Sequence[Sequence[Sequence[float]]]] = None,
# ):
#     v = Validation(dataset)
#     training_errors, testing_errors, average_errors, minimum_errors = \
#         v.grid_search(layer_sizes=layer_sizes, activation=activation, activation_hidden=activation_hidden,
#                       learning_algorithm=learning_algorithm, eta=eta, alpha=alpha, alambd=alambd,
#                       error_calculator=error_calculator, dim_mini_batches=dim_mini_batch, epochs_limit=epochs_limit,
#                       epsilon=epsilon, penalty=penalty, seed=seed, layers=layers)
#     return training_errors, testing_errors, average_errors, minimum_errors
