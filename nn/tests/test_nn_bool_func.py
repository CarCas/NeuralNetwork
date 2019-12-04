from nn.neural_network import ErrorTypes
import unittest
from nn import NeuralNetwork as NN, Architecture, Online, Batch
from nn.activation_function import sigmoidal
import numpy as np
import matplotlib.pyplot as plt


class TestNNBoolFunc(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        self.nn = NN(
            activation_output=sigmoidal,
            learning_algorithm=Batch(),
            eta=0.9,
            max_epochs=10000,
            architecture=Architecture(
                size_input_nodes=2,
                size_output_nodes=1,
                size_hidden_nodes=5,
                range_weights=0.5,
                threshold=1
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
        self.nn().train(data, error_type=ErrorTypes.MIS)

        self.assertEqual(self.nn.compute_error(data), 0)

        plt.plot(self.nn.training_errors)
        plt.show()


if __name__ == '__main__':
    unittest.main()
