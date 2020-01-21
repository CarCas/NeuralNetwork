import unittest

from nn import sigmoid, MultilayerPerceptron, NeuralNetwork, minibatch
import numpy as np


class TestMiniBatch(unittest.TestCase):
    def test_batch_explicit(self):
        np.random.seed(0)

        nn = NeuralNetwork(
            architecture=MultilayerPerceptron(
                layers=[[[0, 1.5, 2], [0, 3, 0.5]],
                        [[0, -1.5, 1.5], [0, -0.5, 2]]],
                alambd=0,
                alpha=0,
                eta=0.5,
                activation=sigmoid,
                activation_hidden=sigmoid,
            ),
            learning_algorithm=minibatch(.5)
        )

        nn.fit([([1, 1], [0, 1]), ([2, 2], [1, 1])])

        nn = nn._current_network
        np.testing.assert_array_almost_equal(
            nn.layers[0],
            [[-9.153689e-05, 1.499817e+00, 1.999817e+00], [1.101478e-04, 3.000220e+00, 5.002203e-01]]
        )

        np.testing.assert_array_almost_equal(
            nn.layers[1],
            [[0.0625, -1.437557, 1.562443], [0.013631, -0.486381, 2.013619]]
        )


if __name__ == '__main__':
    unittest.main()
