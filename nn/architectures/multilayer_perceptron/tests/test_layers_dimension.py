from nn.activation_functions import identity
import unittest
import numpy as np

from nn import sigmoid, identity
from nn.architectures.multilayer_perceptron.architecture import MultilayerPerceptron
from nn.architectures.multilayer_perceptron.neural_network import MLPNeuralNetwork


class TestLayersDimension(unittest.TestCase):
    def test_layers_dimension(self):
        np.random.seed(seed=0)

        nn = MultilayerPerceptron(3, 1, 4, 10)(activation=identity, activation_hidden=sigmoid, eta=0.5)
        np.testing.assert_equal(len(nn.layers), 3)

        np.testing.assert_equal(len(nn.layers[0]), 1)
        for w in nn.layers[0]:
            np.testing.assert_equal(len(w), 4)

        np.testing.assert_equal(len(nn.layers[1]), 4)
        for w in nn.layers[1]:
            np.testing.assert_equal(len(w), 2)

        np.testing.assert_equal(len(nn.layers[2]), 10)
        for w in nn.layers[2]:
            np.testing.assert_equal(len(w), 5)

        nn([1.5, 2, 2.5])


if __name__ == '__main__':
    unittest.main()
