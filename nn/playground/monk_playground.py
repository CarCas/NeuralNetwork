from nn.learning_algorithm import LearningAlgorithm
from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, online, minibatch, batch, relu
from nn import ErrorCalculator
from nn.activation_functions import identity, tanh, tanh_classification
from nn.playground.utilities import encode_categorical, plot
from nn.playground.utilities import read_monk
import numpy as np


if __name__ == '__main__':
    train_data, test_data = read_monk(1)

    nn = NN(
        seed=0,
        epochs_limit=10,
        learning_algorithm=batch,
        n_init=10,
        error_calculator=ErrorCalculator.MSE,
        architecture=MultilayerPerceptron(
            2,
            eta=0.8,
            alpha=0.8,
            alambd=0,
            activation=tanh_classification,
            activation_hidden=relu,
        ),
    )

    nn.fit(train_data)

    nn.error_calculator = ErrorCalculator.ACC
    print('acc', nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MSE
    print('mse', nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MEE
    print('mee', nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MSE
    training_error = nn.compute_learning_curve(train_data)
    testing_error = nn.compute_learning_curve(test_data)
    plot(training_error, testing_error)
