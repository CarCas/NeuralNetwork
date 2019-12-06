import unittest
import numpy as np

from nn import sigmoid, identity
from nn.architectures.multilayer_perceptron.neural_network import MLPMatrix


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = MLPMatrix(2, 2, 2, eta=0.5, activation=identity, activation_hidden=sigmoid)
        self.nn.layers[0] = np.array([[0.35, 0.15, 0.2], [0.35, 0.25, 0.3]])
        self.nn.layers[1] = np.array([[0.6, 0.4, 0.45], [0.6, 0.5, 0.55]])

    def test_feed_forward(self):
        nn = self.nn

        np.testing.assert_array_equal(
            nn([1, 1],
               [2, 10],
               [0.05, 0.1]),
            [[1.1872023850485183, 1.3251161125278352],
             [1.4142280411701063, 1.6055455057691164],
             [1.1059059670597702, 1.2249214040964653]])

    def test_backpropagation(self):
        nn = self.nn
        nn.train([([1, 1], [1, 1])])

        np.testing.assert_array_almost_equal(
            nn([1, 1],
               [2, 10],
               [0.05, 0.1]),
            [[0.9920183389327484, 0.9935135065228662],
             [1.1855227423664705, 1.2150898086359008],
             [0.9301317572876343, 0.9228152582712027]])

    def test_batch(self):
        nn = self.nn
        nn.train([([1, 1], [1, 1])])

        np.testing.assert_array_almost_equal(
            nn([1, 1],
               [2, 10],
               [0.05, 0.1]),
            [[0.9920183389327484, 0.9935135065228662],
             [1.1855227423664705, 1.2150898086359008],
             [0.9301317572876343, 0.9228152582712027]])


if __name__ == '__main__':
    unittest.main()
