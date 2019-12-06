from nn.architectures import multilayer_perceptron
import unittest
import numpy as np

from nn import MultilayerPerceptron, sigmoid
from nn.architectures.multilayer_perceptron.neural_network import MLPMatrix


class TestGlorotBengio(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=1)

        self.nn: MLPMatrix = MultilayerPerceptron(200, 1000, 500)(sigmoid, sigmoid, eta=0.5)
        output_layer = self.nn.layers[-1]
        self.hidden_layer = self.nn.layers[0]

        self.output_w = output_layer.T[1:].T
        self.hidden_w = self.hidden_layer.T[1:].T

        self.len_input = len(self.hidden_layer[0]) - 1

    def test_bias_equal_0(self):
        self.assertFalse(self.hidden_layer.T[0].any())

    def test_min_value_output(self):
        self.assertTrue((self.output_w > -1/np.sqrt(len(self.hidden_layer))).all())
        self.assertFalse((self.output_w > 0.1+-1/np.sqrt(len(self.hidden_layer))).all())

    def test_min_value_hidden(self):
        self.assertTrue((self.hidden_w > -1/np.sqrt(self.len_input)).all())
        self.assertFalse((self.hidden_w > 0.1+-1/np.sqrt(self.len_input)).all())

    def test_max_value_output(self):
        self.assertTrue((self.output_w < 1/np.sqrt(len(self.hidden_layer))).all())
        self.assertFalse((self.output_w < -0.1+1/np.sqrt(len(self.hidden_layer))).all())

    def test_max_value_hidden(self):
        self.assertTrue((self.hidden_w < 1/np.sqrt(self.len_input)).all())
        self.assertFalse((self.hidden_w < -0.1+1/np.sqrt(self.len_input)).all())


if __name__ == '__main__':
    unittest.main()
