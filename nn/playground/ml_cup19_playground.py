from copy import deepcopy
from timeit import timeit

from nn.neural_network import NeuralNetwork
from nn import MultilayerPerceptron, sigmoid, identity, batch, online, ErrorCalculator

from nn.playground.utilities import read_ml_cup_tr, plot


if __name__ == '__main__':
    data = read_ml_cup_tr()
    input_size = len(data[0][0])
    output_size = len(data[0][1])

    nn = NeuralNetwork(
        seed=0,
        architecture=MultilayerPerceptron(
            input_size, 100, output_size,
            activation=identity,
            eta=0.01,
            alpha=0.1,
            alambd=0.01
        ),
        epochs_limit=1000,
    )
    # nn_internal = deepcopy(nn.internal_network)

    print(timeit(lambda: nn.train(data), number=1))
    # print(timeit(lambda: [nn_internal.train(data) for _ in range(nn.epochs_limit)], number=1))

    errors = nn.compute_learning_curve(data)

    plot(errors)

    print(errors[-1])
