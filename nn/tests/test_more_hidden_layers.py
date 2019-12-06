import unittest
import numpy as np

from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron
from nn.architectures.multilayer_perceptron.neural_network import MLPMatrix


class TestMultilayerPerceptronWithMoreHiddenLayer(unittest.TestCase):
    def setUp(self):
        self.nn: MLPMatrix = MLPMatrix(
            2, 2, 3, 2,
            eta=0.5,
            activation=sigmoid,
            activation_hidden=sigmoid)

        self.nn.layers[0] = np.array([[0.3, 0.1, 0.4], [0.45, 0.2, 0.2], [0.7, 0.2, 0.3]])
        self.nn.layers[1] = np.array([[0.35, 0.15, 0.2, 0.12], [0.3, 0.25, 0.3, 0.4]])
        self.nn.layers[2] = np.array([[0.6, 0.4, 0.45], [0.6, 0.5, 0.55]])

    def test_feed_forward(self):
        nn = self.nn

        np.testing.assert_array_equal(nn([1, 1]), [[0.7674756500864669, 0.7914232737978808]])

    def test_backpropagation(self):
        nn = self.nn
        nn.train([([1, 1], [1, 1])])

        np.testing.assert_array_equal(nn([1, 1]), [[0.7750152745318835, 0.7973299193713835]])


if __name__ == '__main__':
    unittest.main()
