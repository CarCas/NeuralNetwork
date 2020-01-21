from nn.learning_algorithm import LearningAlgorithm
from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, online, minibatch, batch, relu
from nn import ErrorCalculator
from nn.activation_functions import identity
from nn.playground.utilities import encode_categorical, plot
from nn.playground.utilities import read_monk
import numpy as np


if __name__ == '__main__':
    train_data, test_data = read_monk(1)

    nn = NN(
        # seed=1,
        epochs_limit=500,
        learning_algorithm=batch,
        n_init=10,
        architecture=MultilayerPerceptron(
            len(train_data[0][0]), 10, len(test_data[0][0]),
            eta=0.5,
            alpha=0.8,
            alambd=0,
            activation=sigmoid,
            activation_hidden=sigmoid,
        ),
    )

    nn.fit(train_data)

    nn.error_calculator = ErrorCalculator.ACC
    print(nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MSE
    training_error = nn.compute_learning_curve(train_data)
    testing_error = nn.compute_learning_curve(test_data)
    plot(training_error, testing_error)
