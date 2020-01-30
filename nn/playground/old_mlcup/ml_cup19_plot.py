from nn import *
import numpy as np
from nn.playground.utilities import read_ml_cup_tr, plot
from timeit import timeit

if __name__ == '__main__':
    ml_cup_training_dataset = read_ml_cup_tr()
    train_data, test_data = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)
    train_set, validation_set = split_dataset(train_data, percentage=2/3, to_shuffle=True, seed=0)

    # seed = 21
    # epochs_limit = 7000
    learning_algorithm = batch
    seed = 34
    epochs_limit = 17011
    # epochs_limit = 100
    size_hidden_layers = [40, 30]
    activation_hidden = tanh
    eta = 0.007
    alpha = 0.55
    alambd = 1e-05

    nn = NeuralNetwork(
        seed=seed,
        epochs_limit=epochs_limit,
        learning_algorithm=batch,
        n_init=1,
        epsilon=0,
        error_calculator=ErrorCalculator.MEE,
        architecture=MultilayerPerceptron(
            size_hidden_layers=size_hidden_layers,
            eta=eta,
            alpha=alpha,
            alambd=alambd,
            activation=identity,
            activation_hidden=activation_hidden,
        ),
    )

    # nn.fit(train_set)

    # nn.error_calculator = ErrorCalculator.MEE
    # print('mee', nn.compute_error(train_set), nn.compute_error(validation_set), nn.compute_error(test_data))

    # training_curve = nn.compute_learning_curve(train_set)
    # validation_curve = nn.compute_learning_curve(validation_set)
    # testing_curve = nn.compute_learning_curve(test_data)

    # plot(training_curve, validation=validation_curve, testing=testing_curve)

    # timeit(lambda: nn.fit(train_set, validation_set, test_data), number=1)
    print(timeit(lambda: nn.fit(train_set, validation_set, test_data), number=1))

    # nn.fit(train_set, validation_set, test_data)

    print('mee', nn.compute_error(train_set), nn.compute_error(validation_set), nn.compute_error(test_data))

    training_curve = nn.training_curve
    validation_curve = nn.validation_curve
    testing_curve = nn.testing_curve

    plot(training_curve, validation=validation_curve, testing=testing_curve)
