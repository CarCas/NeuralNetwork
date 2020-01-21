from copy import deepcopy
from timeit import timeit

from nn.neural_network import NeuralNetwork
from nn import MultilayerPerceptron, sigmoid, identity, batch, online, ErrorCalculator, relu

from nn.playground.utilities import read_ml_cup_tr, plot


if __name__ == '__main__':
    train_data = read_ml_cup_tr()

    nn = NeuralNetwork(
        seed=0,
        architecture=MultilayerPerceptron(
            10,
            activation=identity,
            activation_hidden=relu,
            eta=0.01,
            alpha=0.1,
            alambd=0.01
        ),
        epochs_limit=1000,
    )

    print('time', timeit(lambda: nn.fit(train_data), number=1))

    nn.error_calculator = ErrorCalculator.ACC
    print(nn.compute_error(train_data))

    nn.error_calculator = ErrorCalculator.MSE
    print(nn.compute_error(train_data))

    nn.error_calculator = ErrorCalculator.MSE
    training_error = nn.compute_learning_curve(train_data)
    plot(training_error)
