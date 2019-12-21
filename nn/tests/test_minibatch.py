import unittest

from nn import sigmoid, MultilayerPerceptron, NeuralNetwork, minibatch


class TestMiniBatch(unittest.TestCase):
    def test_batch_explicit(self):
        nn = NeuralNetwork(
            architecture=MultilayerPerceptron(
                layers=[[[0, 1.5, 2], [0, 3, 0.5]],
                        [[0, -1.5, 1.5], [0, -0.5, 2]]],
                eta=0.5,
                activation=sigmoid,
                activation_hidden=sigmoid,
            ),
            learning_algorithm=minibatch(.5)
        )

        nn.train([([1, 1], [0, 1]), ([2, 2], [1, 1])])

        """
        self.assertTrue(np.isclose(
            nn.layers[1],
            [[0., -1.49911246, 1.50088754], [0.01406306, -0.48615559, 2.01384441]]
        ).all())

        self.assertTrue(np.isclose(
            nn.layers[0],
            [[1.18486022e-03, 1.50113909e+00, 2.00113909e+00], [-8.66234240e-04, 2.99918884e+00, 4.99188840e-01]]
        ).all())
        """


if __name__ == '__main__':
    unittest.main()
