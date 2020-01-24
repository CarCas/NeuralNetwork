from nn.learning_algorithm import LearningAlgorithm
from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, online, minibatch, batch, relu, ErrorCalculator, \
    NeuralNetwork
from nn import ErrorCalculator
from nn.activation_functions import identity, tanh, tanh_classification
from nn.playground.utilities import encode_categorical, plot
from nn.playground.utilities import read_monk
import numpy as np
from typing import Mapping, Sequence, Any, Dict
from nn.validation import validation, k_fold_CV, split_dataset, shuffle, grid_search

if __name__ == '__main__':
    # np.random.seed(0)

    train_data, test_data = read_monk(1)

    # nn = NN(
    #     epochs_limit=500,
    #     learning_algorithm=batch,
    #     n_init=20,
    #     error_calculator=ErrorCalculator.MSE,
    #     patience=10,
    #     architecture=MultilayerPerceptron(
    #         2,
    #         eta=0.5,
    #         alpha=0.8,
    #         alambd=0,
    #         activation=sigmoid,
    #         activation_hidden=relu,
    #     )
    # )

    # print('len_training', len(train_data))

    # print('validation', validation(nn.set(), train_data, test_data, error_calculator=ErrorCalculator.ACC))

    # shuffled_train_data = shuffle(train_data)
    # splitted_train_set, splitted_valid_set = split_dataset(shuffled_train_data, 9/10)
    # print('dims', len(splitted_train_set), len(splitted_valid_set))
    # print('validation splitted', validation(nn.set(), splitted_train_set, splitted_valid_set,
    #                                         error_calculator=ErrorCalculator.ACC))

    # print('k_fold_CV ', k_fold_CV(nn.set(), train_data, error_calculator=ErrorCalculator.ACC, cv=10, to_shuffle=True))

    # training_error = nn.compute_learning_curve(train_data, ErrorCalculator.ACC)
    # testing_error = nn.compute_learning_curve(test_data, ErrorCalculator.ACC)
    # plot(training_error, testing_error)


################
    params_nn: Dict[str, Sequence[Any]] = dict(
        error_calculator=[ErrorCalculator.MSE],
        learning_algorithm=[batch],
        epochs_limit=[500],
        n_init=[10],
        epsilon=[1e-5],
        patience=[10],
    )
    params_architecture: Mapping[str, Sequence[Any]] = dict(
        size_hidden_layers=[(2,)],
        activation=[sigmoid],
        activation_hidden=[relu],
        eta=[0.3, 0.5, 0.8],
        alpha=[0.3, 0.5, 0.8],
        alambd=[0],
        eta_decay=[0],
        eta_min=[0.01],
    )

    cv_params: Mapping[str, Any] = dict(
        cv=10,
        error_calculator=ErrorCalculator.ACC,
        to_shuffle=True,
    )

    grid_search_results = grid_search(
        train_data,
        params_nn=params_nn,
        params_architecture=params_architecture,
        cv_params=cv_params,

        n_jobs=8
    )

    for i, entry in enumerate(grid_search_results[::-1][:3]):
        for key, value in entry._asdict().items():
            print('{}:'.format(key), value)
