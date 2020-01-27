from timeit import timeit

from nn.neural_network import NeuralNetwork
from nn import *
import numpy as np
from nn.playground.utilities import read_ml_cup_tr, plot


if __name__ == '__main__':
    ml_cup_training_dataset = read_ml_cup_tr()

    train_val_set, test_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)

    train_set, validation_set = split_dataset(train_val_set, percentage=2/3, to_shuffle=True)

    print('dataset', len(ml_cup_training_dataset))
    print('train_set', len(train_set))
    print('validation_set', len(validation_set))
    print('test_set', len(test_set))

    np.random.seed()

    nn = NeuralNetwork(
        # seed=0,
        learning_algorithm=batch,
        architecture=MultilayerPerceptron(
            100,
            activation=identity,
            activation_hidden=tanh,
            eta=0.01,
            alpha=0,
            alambd=0,
            eta_decay=0,
        ),
        error_calculator=ErrorCalculator.MEE,
        n_init=3,
        epochs_limit=2000,
        epsilon=1e-5,
        save_internal_networks=False,
    )

    # print('time', timeit(lambda: nn.fit(train_set), number=1))

    val_result = validation(nn, train_set, validation_set, ErrorCalculator.MEE)

    nn.error_calculator = ErrorCalculator.MEE
    training_curve = nn.compute_learning_curve(train_set)
    validation_curve = nn.compute_learning_curve(validation_set)
    testing_curve = nn.compute_learning_curve(test_set)

    print('val_result', val_result)

    print(testing_curve[val_result.epoch])

    plot(training_curve, validation_curve, testing_curve)
