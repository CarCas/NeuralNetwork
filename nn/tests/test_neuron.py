import unittest

from nn.neuron import Neuron
from nn.activation_function import identity, sign


class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.w = [2, 3, 4]
        self.x = [6, 7]
        self.net_result = 48

        self.and_neuron = Neuron(
            bias=-1.5,
            w=[1, 1],
            activation=sign)

        self.or_neuron = Neuron(
            bias=-0.5,
            w=[1, 1],
            activation=sign)

        self.xor_out_neuron = Neuron(
            bias=-0.5,
            w=[-1, 1],
            activation=sign)

    def test_net(self):
        u = Neuron(bias=self.w[0], w=self.w[1:], activation=identity)
        u(*self.x)
        self.assertEqual(u.net, self.net_result)

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
