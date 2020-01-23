from nn.learning_algorithm import LearningAlgorithm
from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, online, minibatch, batch, relu, ErrorCalculator, \
    NeuralNetwork
from nn import ErrorCalculator
from nn.activation_functions import identity, tanh, tanh_classification
from nn.playground.utilities import encode_categorical, plot
from nn.playground.utilities import read_monk
import numpy as np
from nn.validation import validation, k_fold_CV, split_dataset, shuffle, grid_search

if __name__ == '__main__':
    np.random.seed(0)

    train_data, test_data = read_monk(1)

    nn = NN(
        # seed=0,
        epochs_limit=500,
        learning_algorithm=batch,
        n_init=20,
        error_calculator=ErrorCalculator.MSE,
        architecture=MultilayerPerceptron(
            2,
            eta=0.5,
            alpha=0.8,
            alambd=0,
            activation=sigmoid,
            activation_hidden=relu)
    )
    """
    print('len_training', len(train_data))

    print('validation', validation(nn.set(), train_data, test_data, error_calculator=ErrorCalculator.ACC))

    shuffled_train_data = shuffle(train_data)
    splitted_train_set, splitted_valid_set = split_dataset(shuffled_train_data, 9/10)
    print('dims', len(splitted_train_set), len(splitted_valid_set))
    print('validation splitted', validation(nn.set(), splitted_train_set, splitted_valid_set,
                                            error_calculator=ErrorCalculator.ACC))

    print('k_fold_CV ', k_fold_CV(nn.set(), train_data, error_calculator=ErrorCalculator.ACC, cv=10, to_shuffle=True))

    training_error = nn.compute_learning_curve(train_data, ErrorCalculator.ACC)
    testing_error = nn.compute_learning_curve(test_data, ErrorCalculator.ACC)
    plot(training_error, testing_error)
    """

    params_ar = {
        'size_hidden_layers': [[2], [2, 2]],
        'eta': [0.3, 0.5, 0.65],
        'alambd': [0, 0.01, 0.001],
        'alpha': [0.5, 0.2, 0.8],
        'activation': [sigmoid],
    }
    params_nnn = {
        'learning_algorithm': [batch, online]
    }

    funcall = grid_search(NeuralNetwork, train_data, params_nn=params_nnn, params_architecture=params_ar,
                          cv_params=dict(cv=10))

    print(funcall[0])
