from typing import List, Optional, Sequence, Tuple

from nn.types import Architecture, ActivationFunction, Pattern
from nn.learning_algorithm import LearningAlgorithm
from nn.error_calculator import ErrorCalculator
from nn import NeuralNetwork, MultilayerPerceptron, sigmoid


class Validation:
    idx: int = 0

    def __init__(self,
                 training: Sequence[Pattern],
                 testing: Sequence[Pattern]):
        self.architecture: List[Architecture] = []
        self.training_set = training
        self.testing_set = testing
        self.training_errors: List[Tuple[int, Sequence[float]]] = []
        self.testing_errors: List[Tuple[int, Sequence[float]]] = []

    def nn_calling(self, architecture: Architecture, learning_algorithm: LearningAlgorithm,
                   error_calculator: ErrorCalculator, penalty: float,
                   epochs_limit: int, epsilon: float, seed: Optional[int], dim_mini_batch: int = -1):
        nn = NeuralNetwork(architecture=architecture, error_calculator=error_calculator,
                           learning_algorithm=learning_algorithm, penalty=penalty,
                           epochs_limit=epochs_limit, epsilon=epsilon, seed=seed, dim_mini_batch=dim_mini_batch)
        nn.train(self.training_set, self.testing_set)
        train_errs = nn.compute_learning_curve(self.training_set)
        test_errs = nn.compute_learning_curve(self.testing_set)
        global idx
        self.training_errors.append((idx, train_errs))
        self.testing_errors.append((idx, test_errs))
        idx += 1

    @staticmethod
    def architecture_create(
            layer_sizes: Tuple[int],
            activation: ActivationFunction,
            activation_hidden: ActivationFunction = sigmoid,
            eta: float = 0.5,
            alpha: float = 0,
            alambd: float = 0.1,

            layers: Optional[Sequence[Sequence[Sequence[float]]]] = None
    ) -> Architecture:
        mp = MultilayerPerceptron(*layer_sizes, activation=activation, activation_hidden=activation_hidden,
                                  eta=eta, alpha=alpha, alambd=alambd, layers=layers)
        return mp

    def list_archs(
            self,
            layer_sizes: List[Tuple[int]],
            activation: List[ActivationFunction],
            activation_hidden: List[ActivationFunction] = sigmoid,
            eta: List[float] = 0.5,
            alpha: List[float] = 0,
            alambd: List[float] = 0.1,

            layers: Optional[Sequence[Sequence[Sequence[float]]]] = None
    ) -> List[Architecture]:
        archs = [self.architecture_create(layer_sizes=ls, activation=act, activation_hidden=ah,
                                          eta=e, alpha=a, alambd=lmb, layers=layers)
                 for ls in layer_sizes
                 for act in activation
                 for ah in activation_hidden
                 for e in eta
                 for a in alpha
                 for lmb in alambd]
        return archs

    def grid_search(self, layer_sizes: List[Tuple[int]], activation: List[ActivationFunction],
                    activation_hidden: List[ActivationFunction], learning_algorithm: List[LearningAlgorithm],
                    error_calculator: List[ErrorCalculator], penalty: List[float], eta: List[float],
                    epochs_limit: List[int], epsilon: List[float], alpha: List[float], alambd: List[float],
                    dim_mini_batch: List[int], seed: List[Optional[int]],
                    layers: Optional[Sequence[Sequence[Sequence[float]]]] = None):
        self.architecture.extend(self.list_archs(layer_sizes=layer_sizes, activation=activation,
                                                 activation_hidden=activation_hidden, eta=eta, alpha=alpha,
                                                 alambd=alambd, layers=layers))
        for arch in self.architecture:
            for la in learning_algorithm:
                for ec in error_calculator:
                    for p in penalty:
                        for el in epochs_limit:
                            for eps in epsilon:
                                for s in seed:
                                    for mb in dim_mini_batch:
                                        self.nn_calling(arch, la, ec, p, el, eps, s, mb)

        return self.training_errors, self.testing_errors


def validate(
        train_set: Sequence[Pattern],
        test_set: Sequence[Pattern],
        layer_sizes: List[Tuple[int]],
        activation: List[ActivationFunction],
        learning_algorithm: List[LearningAlgorithm],
        error_calculator: List[ErrorCalculator],
        dim_mini_batch: List[int],
        epochs_limit: List[int],
        epsilon: List[float],
        penalty: List[float],
        seed: List[Optional[int]],
        activation_hidden: List[ActivationFunction] = sigmoid,
        eta: List[float] = 0.5,
        alpha: List[float] = 0,
        alambd: List[float] = 0.1,

        layers: Optional[Sequence[Sequence[Sequence[float]]]] = None,
):
    v = Validation(train_set, test_set)
    training_errors, testing_errors =\
        v.grid_search(layer_sizes=layer_sizes, activation=activation, activation_hidden=activation_hidden,
                      learning_algorithm=learning_algorithm, eta=eta, alpha=alpha, alambd=alambd,
                      error_calculator=error_calculator, dim_mini_batch=dim_mini_batch, epochs_limit=epochs_limit,
                      epsilon=epsilon, penalty=penalty, seed=seed, layers=layers)
