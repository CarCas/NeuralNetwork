import unittest
from nn import NeuralNetwork as NN
from nn.activation_function import sigmoidal
import random


class TestNNBoolFunc(unittest.TestCase):
    def setUp(self):
        random.seed(1)

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
            architecture=NN.Architecture(
                number_inputs=2,
                number_outputs=1,
                number_hidden=5
            ))

        error = 1
        while(error):
            nn.train(data, 1, 1)
            error = 0
            for x, d in data:
                error += (round(nn(*x)[0]) - d[0])**2

        self.assertEqual(error, 0)


if __name__ == '__main__':
    unittest.main()
