from nn.learning_algorithm import LearningAlgorithm
from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, online, minibatch, batch, relu
from nn import ErrorCalculator
from nn.activation_functions import identity
from nn.playground.utilities import encode_categorical, plot, encode_categorical2
from nn.playground.utilities import (
    read_monk_1_tr,
    read_monk_1_ts,
    read_monk_2_tr,
    read_monk_2_ts,
    read_monk_3_tr,
    read_monk_3_ts,
)
import numpy as np


if __name__ == '__main__':
    train_data = read_monk_3_tr()
    test_data = read_monk_3_ts()

    train_data1, test_data1 = encode_categorical(train_data, test_data)
    train_data2, test_data2 = encode_categorical2(train_data, test_data)

    print(len(train_data1[0][0]), len(train_data1[0][0]))

    for i, _ in enumerate(train_data1):
        if train_data1[i] == train_data2[i]:
            print("Sbagliato! Wrong!!")
            break
    print(":D")

    print(1)
    for x in train_data1[:10]:
        print(x)
    print(2)
    for x in train_data2[:10]:
        print(x)

    train_data, test_data = train_data2, test_data2

    nn = NN(
        # seed=4,
        epochs_limit=500,
        learning_algorithm=batch,
        architecture=MultilayerPerceptron(
            len(train_data[0][0]), 10, len(test_data[0][0]),
            eta=0.5,
            alpha=0.8,
            alambd=0,
            activation=sigmoid,
            activation_hidden=sigmoid,
        ),
    )

    # w = nn.internal_network.layers

    # nn.train(train_data[:2])
    nn.train(train_data)

    # print('deltaOutput_', nn.internal_network._deltas[-1])
    # print('deltaHidden_', nn.internal_network._deltas[-2])

    # print()

    # for l in w: print(repr(l.T))

    # nn.error_calculator = ErrorCalculator.MIS
    # print(nn.compute_error(train_data), nn.compute_error(test_data))

    # nn.error_calculator = ErrorCalculator.MIS
    # training_error = nn.compute_learning_curve(train_data)
    # testing_error = nn.compute_learning_curve(test_data)
    # plot(training_error, testing_error, False)

    nn.error_calculator = ErrorCalculator.ACC
    print(nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MSE
    training_error = nn.compute_learning_curve(train_data)
    testing_error = nn.compute_learning_curve(test_data)
    plot(training_error, testing_error)

    # nn.error_calculator = ErrorCalculator.MIS
    # print(nn.compute_error(train_data), nn.compute_error(test_data))

