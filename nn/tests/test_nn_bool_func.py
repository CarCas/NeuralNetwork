import unittest
from nn import NeuralNetwork as NN, Architecture
from nn.activation_function import sigmoidal
import numpy as np


class TestNNBoolFunc(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        pass

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
        nn = NN(
            activation=sigmoidal,
            architecture=Architecture(
                size_input_nodes=2,
                size_output_nodes=1,
                size_hidden_nodes=5
            ))

        error = 1
        while(error):
            nn.train(data, [])
            error = 0
            for x, d in data:
                error += (round(nn(*x)[0]) - d[0])**2

            # print(nn.test(data), error)

        self.assertEqual(error, 0)


if __name__ == '__main__':
    unittest.main()
