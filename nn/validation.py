from typing import List, Optional, Sequence

from nn.types import Architecture, ActivationFunction, Pattern
from nn.learning_algorithm import LearningAlgorithm
from nn.error_calculator import ErrorCalculator
from nn import NeuralNetwork


class Validation:

    def __init__(self, arch: List[Architecture], training: Sequence[Pattern], testing: Sequence[Pattern]):
        self.architecture: List[Architecture] = []
        self.architecture.extend(arch)
        self.training_set = training
        self.testing_set = testing
        self.training_errors = []
        self.testing_errors = []

    def nn_calling(self, architecture: Architecture, activation: ActivationFunction,
                   activation_hidden: ActivationFunction, learning_algorithm: LearningAlgorithm,
                   error_calculator: ErrorCalculator, penalty: float, eta: float,
                   epochs_limit: int, epsilon: float, alpha: float, seed: Optional[int]):
        nn = NeuralNetwork(activation=activation, architecture=architecture, activation_hidden=activation_hidden,
                           error_calculator=error_calculator, learning_algorithm=learning_algorithm,
                           penalty=penalty, eta=eta, epochs_limit=epochs_limit, epsilon=epsilon, seed=seed, alpha=alpha)
        nn.train(self.training_set, self.testing_set)
        train_errs = nn.compute_learning_curve(self.training_set)
        test_errs = nn.compute_learning_curve(self.testing_set)
        self.training_errors.append(train_errs)
        self.testing_errors.append(test_errs)

    def grid_search(self, activation: List[ActivationFunction], activation_hidden: List[ActivationFunction],
                    learning_algorithm: List[LearningAlgorithm], error_calculator: List[ErrorCalculator],
                    penalty: List[float], eta: List[float], epochs_limit: List[int], epsilon: List[float],
                    seed: List[Optional[int]]):
        return [
            self.nn_calling(arch, a, ah, la, ec, p, e, el, eps, s)
            for arch in self.architecture
            for a in activation
            for ah in activation_hidden
            for la in learning_algorithm
            for ec in error_calculator
            for p in penalty
            for e in eta
            for eps in epsilon
            for el in epochs_limit
            for s in seed
        ]
