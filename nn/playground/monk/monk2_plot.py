from nn.learning_algorithm import LearningAlgorithm
from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, online, minibatch, batch, relu
from nn import ErrorCalculator
from nn.activation_functions import identity, tanh, tanh_classification
from nn.utilities import encode_categorical, plot
from nn.utilities import read_monk
import numpy as np
from nn import validation, split_dataset


if __name__ == '__main__':
    train_data, test_data = read_monk(2)
    train_set, validation_set = split_dataset(train_data, 2/3, to_shuffle=True, seed=0)

    seed = 23
    epochs_limit = 896
    eta = 0.1
    alpha = 0.6
    alambd = 0
    validation_error = ErrorCalculator.MSE

    nn = NN(
        seed=seed,
        epochs_limit=epochs_limit,
        learning_algorithm=batch,
        n_init=1,
        error_calculator=ErrorCalculator.MSE,
        architecture=MultilayerPerceptron(
            size_hidden_layers=(2,),
            eta=eta,
            alpha=alpha,
            alambd=alambd,
            activation=tanh_classification,
            activation_hidden=relu,
        ),
    )

    nn.fit(train_set)

    nn.error_calculator = ErrorCalculator.MSE
    print('mse', nn.compute_error(train_set), nn.compute_error(validation_set), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MEE
    print('mee', nn.compute_error(train_set), nn.compute_error(validation_set), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.ACC
    print('acc', nn.compute_error(train_set), nn.compute_error(validation_set), nn.compute_error(test_data))

    # MSE
    nn.error_calculator = ErrorCalculator.MSE
    training_curve = nn.compute_learning_curve(train_set)
    validation_curve = nn.compute_learning_curve(validation_set)
    testing_curve = nn.compute_learning_curve(test_data)

    print('mse_last_curve', training_curve[-1], validation_curve[-1], testing_curve[-1])
    plot(training_curve, validation=validation_curve, testing=testing_curve)

    # ACC
    nn.error_calculator = ErrorCalculator.ACC
    training_curve = nn.compute_learning_curve(train_set)
    validation_curve = nn.compute_learning_curve(validation_set)
    testing_curve = nn.compute_learning_curve(test_data)

    print('acc_last_curve', training_curve[-1], validation_curve[-1], testing_curve[-1])
    plot(training_curve, validation=validation_curve, testing=testing_curve)
