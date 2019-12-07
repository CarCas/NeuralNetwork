from nn.neural_network import ErrorCalculator
import unittest
from nn import NeuralNetwork as NN, MultilayerPerceptron, LearningAlgorithm
from nn.activation_functions import sigmoid


class TestNNBoolFunc(unittest.TestCase):
    def setUp(self):
        self.nn = NN(
            seed=0,
            activation=sigmoid,
            eta=0.99,
            learning_algorithm=LearningAlgorithm.ONLINE,
            error_calculator=ErrorCalculator.MIS,
            architecture=MultilayerPerceptron(2, 2, 1))

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

        while self.nn.compute_error(data) > 0:
            self.nn.train(data)

        self.assertEqual(self.nn.compute_error(data), 0)


if __name__ == '__main__':
    unittest.main()
