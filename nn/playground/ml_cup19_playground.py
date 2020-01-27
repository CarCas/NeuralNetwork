from timeit import timeit

from nn.neural_network import NeuralNetwork
from nn import *

from nn.playground.utilities import read_ml_cup_tr, plot


if __name__ == '__main__':

    import warnings
    train_data = read_ml_cup_tr()

    nn = NeuralNetwork(
        seed=0,
        learning_algorithm=minibatch(0.3),
        architecture=MultilayerPerceptron(
            100,
            activation=identity,
            activation_hidden=relu,
            eta=0.01,
            # alpha=0.8,
            # alambd=0.1,
            # eta_decay=0.1
        ),
        error_calculator=ErrorCalculator.MSE,
        n_init=1,
        epochs_limit=100,
        epsilon=1e-3,
        save_internal_networks=False,
    )

    print('time', timeit(lambda: nn.fit(train_data), number=1))

    print(nn.compute_error(train_data))

    training_error = nn.compute_learning_curve(train_data)
    plot(training_error)
