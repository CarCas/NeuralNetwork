from nn.neuron_layer import NeuronLayer
import unittest

from nn.activation_function import sign


class TestNeuronLayer(unittest.TestCase):
    def test_layer(self):
        hidden_layer = NeuronLayer(
            activation=sign,
            weights=[[-1.5, 1, 1], [-0.5, 1, 1]])

        output_layer = NeuronLayer(
            activation=sign,
            weights=[[-0.5, -1, 1]])

        self.assertEqual(output_layer(*hidden_layer(0, 0)), (0,))
        self.assertEqual(output_layer(*hidden_layer(0, 1)), (1,))
        self.assertEqual(output_layer(*hidden_layer(1, 0)), (1,))
        self.assertEqual(output_layer(*hidden_layer(1, 1)), (0,))


if __name__ == '__main__':
    unittest.main()
