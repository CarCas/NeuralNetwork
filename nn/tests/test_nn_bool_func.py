from nn.neural_network import ErrorCalculator
import unittest
from nn import NeuralNetwork as NN, MultilayerPerceptron, batch, identity
from nn.activation_functions import sigmoid, relu


class TestNNBoolFunc(unittest.TestCase):
    def setUp(self):
        self.nn = NN(
            seed=0,
            learning_algorithm=batch,
            error_calculator=ErrorCalculator.MIS,
            architecture=MultilayerPerceptron(
                2,
                activation=sigmoid,
                activation_hidden=relu,
                alambd=0,
                alpha=0.9,
                eta=0.9,
            ))

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
        self.nn.set()

        while True:
            self.nn.fit(data)
            print(self.nn.compute_error(data))
            if self.nn.compute_error(data) == 0:
                break

        self.assertEqual(self.nn.compute_error(data), 0)


if __name__ == '__main__':
    unittest.main()
