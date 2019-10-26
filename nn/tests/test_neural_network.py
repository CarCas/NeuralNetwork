import unittest
import numpy as np

from nn import NeuralNetwork as NN
from nn.activation_function import identity, sigmoidal_1 as sigmoidal


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        pass

    def test_feed_forward(self):
        nn = NN(
            activation=sigmoidal,
            architecture=NN.Architecture(
                number_inputs=2,
                number_outputs=2,
                number_hidden=2,

                hidden_weights=((0.35, 0.15, 0.2), (0.35, 0.25, 0.3)),
                output_weights=((0.6, 0.4, 0.45), (0.6, 0.5, 0.55))
            ))

        self.assertEqual(nn(1, 1),
                         (0.766240339686048, 0.7900316350128183))
        self.assertEqual(nn(2, 10),
                         (0.80443196033845, 0.8327920196423029))
        self.assertEqual(nn(0.05, 0.1),
                         (0.7513650695523157, 0.7729284653214625))

        nn.train([([1, 1], [1, 1])], 1, 0.5)
        self.assertEqual(nn(1, 1),
                         (0.7700458403850474, 0.7930571932467279))
        nn.train([([2, 10], [0.05, 0.1])], 1, 0.5)
        self.assertEqual(nn(1, 1),
                         (0.7552637045019276, 0.7810102555963433))


if __name__ == '__main__':
    unittest.main()
