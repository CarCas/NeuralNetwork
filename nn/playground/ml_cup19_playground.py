from copy import deepcopy
from timeit import timeit

from nn.neural_network import NeuralNetwork
from nn import MultilayerPerceptron, sigmoid, identity, batch, online

from nn.playground.utilities import read_ml_cup_tr, plot


if __name__ == '__main__':
    data = read_ml_cup_tr()
    input_size = len(data[0][0])
    output_size = len(data[0][1])

    nn = NeuralNetwork(
        seed=0,
        activation=identity,
        eta=0.05,
        architecture=MultilayerPerceptron(input_size, 1, output_size),
        epochs_limit=100
    )
    nn_internal = deepcopy(nn.internal_network)

    print(timeit(lambda: nn.train(data), number=1))
    print(timeit(lambda: [nn_internal.train(data) for _ in range(nn.epochs_limit)], number=1))

    plot(nn.compute_learning_curve(data))
