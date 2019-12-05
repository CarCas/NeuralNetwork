import unittest
import numpy as np

from nn import sigmoid, MultilayerPerceptron, Online


class TestGlorotBengio(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        self.nn = MultilayerPerceptron(
            learining_algorthm=Online(),
            size_input_nodes=200,
            size_output_nodes=500,
            size_hidden_nodes=1000,
        )(activation=sigmoid)

        self.output_w = np.array(self.nn.output_layer.w).T[1:].T
        self.hidden_w = np.array(self.nn.hidden_layer.w).T[1:].T

    def test_bias_equal_0(self):
        self.assertFalse(np.array(self.nn.hidden_layer.w).T[0].any())

    def test_min_value_output(self):
        self.assertTrue((self.output_w > -1/np.sqrt(len(self.nn.hidden_layer.w))).all())
        self.assertFalse((self.output_w > 0.1+-1/np.sqrt(len(self.nn.hidden_layer.w))).all())

    def test_min_value_hidden(self):
        self.assertTrue((self.hidden_w > -1/np.sqrt(len(self.nn.input))).all())
        self.assertFalse((self.hidden_w > 0.1+-1/np.sqrt(len(self.nn.input))).all())

    def test_max_value_output(self):
        self.assertTrue((self.output_w < 1/np.sqrt(len(self.nn.hidden_layer.w))).all())
        self.assertFalse((self.output_w < -0.1+1/np.sqrt(len(self.nn.hidden_layer.w))).all())

    def test_max_value_hidden(self):
        self.assertTrue((self.hidden_w < 1/np.sqrt(len(self.nn.input))).all())
        self.assertFalse((self.hidden_w < -0.1+1/np.sqrt(len(self.nn.input))).all())


if __name__ == '__main__':
    unittest.main()
