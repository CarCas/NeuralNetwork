from nn.learning_algorithm import LearningAlgorithm
from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, online, minibatch, batch, relu
from nn import ErrorCalculator
from nn.activation_functions import identity, tanh, tanh_classification
from nn.playground.utilities import encode_categorical, plot
from nn.playground.utilities import read_monk
import numpy as np
from nn import validation, split_dataset


if __name__ == '__main__':
    train_data, test_data = read_monk(3)

    layers = (4,)
    eta = 0.01
    alpha = 0.9
    alambd = 0

    nn = NN(
        seed=0,
        epochs_limit=5000,
        learning_algorithm=batch,
        n_init=10,
        error_calculator=ErrorCalculator.MSE,
        architecture=MultilayerPerceptron(
            size_hidden_layers=layers,
            eta=eta,
            alpha=alpha,
            alambd=alambd,
            activation=tanh_classification,
            activation_hidden=relu,
        ),
    )

    train_set, validation_set = split_dataset(train_data, 2/3, to_shuffle=True)

    val_result = validation(nn, train_set, validation_set, ErrorCalculator.ACC)

    print(val_result)

    nn = NN(
        seed=0,
        epochs_limit=val_result.epoch,
        learning_algorithm=batch,
        n_init=10,
        error_calculator=ErrorCalculator.MSE,
        architecture=MultilayerPerceptron(
            size_hidden_layers=layers,
            eta=eta,
            alpha=alpha,
            alambd=alambd,
            activation=tanh_classification,
            activation_hidden=relu,
        ),
    )

    nn.fit(train_data)

    nn.error_calculator = ErrorCalculator.MSE
    print('mse', nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MEE
    print('mee', nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.ACC
    print('acc', nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MSE
    training_curve = nn.compute_learning_curve(train_data)
    testing_curve = nn.compute_learning_curve(test_data)
    plot(training_curve, testing=testing_curve, title='monk3 - MEE')

    nn.error_calculator = ErrorCalculator.ACC
    training_curve = nn.compute_learning_curve(train_data)
    testing_curve = nn.compute_learning_curve(test_data)
    plot(training_curve, testing=testing_curve, title='monk3 - accuracy')
