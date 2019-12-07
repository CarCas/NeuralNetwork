from copy import deepcopy
from timeit import timeit

from nn.neural_network import NeuralNetwork
from nn import MultilayerPerceptron, sigmoid, identity, LearningAlgorithm

from nn.playground.utilities import read_ml_cup_tr, plot


if __name__ == '__main__':
    data = read_ml_cup_tr()
    input_size = len(data[0][0])
    output_size = len(data[0][1])

    n1 = NeuralNetwork(
        seed=0,
        activation=identity,
        eta=0.05,
        architecture=MultilayerPerceptron(input_size, 10, output_size),
        epochs_limit=100
    )
    n2 = deepcopy(n1)

    print(timeit(lambda: n1.train(data), number=1))
    print(timeit(lambda: [n2.internal_network.train(data) for _ in range(n2.epochs_limit)], number=1))

    plot(n1.compute_learning_curve(data))
