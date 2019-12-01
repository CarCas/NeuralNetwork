import unittest

from nn.neuron import Neuron
from nn.activation_function import identity, sign
import numpy as np


class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.and_neuron = Neuron(
            w=[-1.5, 1, 1],
            activation=sign)

        self.or_neuron = Neuron(
            w=[-0.5, 1, 1],
            activation=sign)

        self.xor_out_neuron = Neuron(
            w=[-0.5, -1, 1],
            activation=sign)

    def test_net(self):
        w = [2, 3, 4]
        x = [5, 6, 7]
        net_result = 1 + np.dot(w, x)

        u = Neuron(w=[1]+w, activation=identity)
        u(*x)
        self.assertEqual(u.net, net_result)

    def test_and(self):
        self.assertEqual(self.and_neuron(0, 0), 0)
        self.assertEqual(self.and_neuron(0, 1), 0)
        self.assertEqual(self.and_neuron(1, 0), 0)
        self.assertEqual(self.and_neuron(1, 1), 1)

    def test_or(self):
        self.assertEqual(self.or_neuron(0, 0), 0)
        self.assertEqual(self.or_neuron(0, 1), 1)
        self.assertEqual(self.or_neuron(1, 0), 1)
        self.assertEqual(self.or_neuron(1, 1), 1)

    def test_out_xor_neuron(self):
        self.assertEqual(self.xor_out_neuron(0, 0), 0)
        self.assertEqual(self.xor_out_neuron(0, 1), 1)
        self.assertEqual(self.xor_out_neuron(1, 0), 0)
        self.assertEqual(self.xor_out_neuron(1, 1), 0)


if __name__ == '__main__':
    unittest.main()
