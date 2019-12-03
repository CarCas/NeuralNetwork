from nn.neural_network import ErrorTypes
import unittest
from nn import NeuralNetwork as NN, Architecture
from nn.activation_function import sigmoidal
import numpy as np


class TestNNBoolFunc(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        self.nn = NN(
            activation=sigmoidal,
            architecture=Architecture(
                size_input_nodes=2,
                size_output_nodes=1,
                size_hidden_nodes=5
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
        self.nn().train(data, error_type=ErrorTypes.MIS, eta=0.75, epoches=1000)

        self.assertEqual(self.nn.compute_error(data), 0)


if __name__ == '__main__':
    unittest.main()
