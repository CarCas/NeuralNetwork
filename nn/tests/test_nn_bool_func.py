from nn.neural_network import ErrorTypes
import unittest
from nn import NeuralNetwork as NN, MultilayerPerceptron
from nn.activation_function import sigmoid


class TestNNBoolFunc(unittest.TestCase):
    def setUp(self):
        self.nn = NN(
            seed=1,
            activation=sigmoid,
            eta=0.9,
            epochs_limit=10000,
            error_types=[ErrorTypes.MIS],
            architecture=MultilayerPerceptron(
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
        self.nn.set().train(data)

        self.assertEqual(self.nn.compute_error(data), 0)


if __name__ == '__main__':
    unittest.main()
