import unittest
from nn import NeuralNetwork as NN, Architecture
from nn.activation_function import sigmoidal
import numpy as np


class TestGlorotBengio(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        self.nn = NN(
            activation_hidden=sigmoidal,
            architecture=Architecture(
                size_input_nodes=200,
                size_output_nodes=500,
                size_hidden_nodes=1000,
            ))

        self.output_w = np.array(self.nn.output_layer.w).T[1:].T
        self.hidden_w = np.array(self.nn.hidden_layer.w).T[1:].T

    def test_bias_equal_0(self):
        self.assertFalse(np.array(self.nn.hidden_layer.w).T[0].any())

    def test_min_value_output(self):
        self.assertTrue((self.output_w > -1/np.sqrt(len(self.nn.hidden_layer.w))).all())
        self.assertFalse((self.output_w > 0.1+-1/np.sqrt(len(self.nn.hidden_layer.w))).all())

    def test_min_value_hidden(self):
        self.assertTrue((self.hidden_w > -1/np.sqrt(len(self.nn.input_layer))).all())
        self.assertFalse((self.hidden_w > 0.1+-1/np.sqrt(len(self.nn.input_layer))).all())

    def test_max_value_output(self):
        self.assertTrue((self.output_w < 1/np.sqrt(len(self.nn.hidden_layer.w))).all())
        self.assertFalse((self.output_w < -0.1+1/np.sqrt(len(self.nn.hidden_layer.w))).all())

    def test_max_value_hidden(self):
        self.assertTrue((self.hidden_w < 1/np.sqrt(len(self.nn.input_layer))).all())
        self.assertFalse((self.hidden_w < -0.1+1/np.sqrt(len(self.nn.input_layer))).all())


if __name__ == '__main__':
    unittest.main()
