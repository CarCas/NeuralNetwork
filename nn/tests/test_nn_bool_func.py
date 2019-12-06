from nn.neural_network import ErrorTypes
import unittest
from nn import NeuralNetwork as NN, MultilayerPerceptron
from nn.activation_functions import sigmoid


class TestNNBoolFunc(unittest.TestCase):
    def setUp(self):
        self.nn = NN(
            seed=1,
            activation=sigmoid,
            eta=0.9,
            epsilon=0,
            epochs_limit=10000,
            error_types=[ErrorTypes.MIS],
            architecture=MultilayerPerceptron(2, 5, 1))

    def test_and(self):
        self.try_data([
            ([0, 0], [0]),
            ([0, 1], [0]),
            ([1, 0], [0]),
            ([1, 1], [1]),
        ])

    def test_or(self):
        self.try_data([
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [1]),
        ])

    def test_xor(self):
        self.try_data([
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ])

    def try_data(self, data):
        self.nn.set().train(data)

        self.assertEqual(self.nn.compute_error(data), 0)


if __name__ == '__main__':
    unittest.main()
