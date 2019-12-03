import unittest
import numpy as np

from nn import NeuralNetwork as NN, Architecture
from nn.activation_function import sigmoidal


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NN(
            activation=sigmoidal,
            architecture=Architecture(
                size_input_nodes=2,
                size_output_nodes=2,
                size_hidden_nodes=2,
                hidden_weights=[[0.35, 0.15, 0.2], [0.35, 0.25, 0.3]],
                output_weights=[[0.6, 0.4, 0.45], [0.6, 0.5, 0.55]]
            ))

    def test_feed_forward(self):
        nn = self.nn

        np.testing.assert_array_equal(nn(1, 1),
                                      [0.766240339686048, 0.7900316350128183])
        np.testing.assert_array_equal(nn(2, 10),
                                      [0.80443196033845, 0.8327920196423029])
        np.testing.assert_array_equal(nn(0.05, 0.1),
                                      [0.7513650695523157, 0.7729284653214625])

    def test_backpropagation(self):
        nn = self.nn
        nn.train([([1, 1], [1, 1])], [], eta=0.5)

        # neural network expected after train
        ne = NN(
            activation=sigmoidal,
            architecture=Architecture(
                size_input_nodes=2,
                size_output_nodes=2,
                size_hidden_nodes=2,
                output_weights=[[0.62093506, 0.41398855, 0.46488377],
                                [0.61741495, 0.51163646, 0.56238115]],
                hidden_weights=[[0.35378719, 0.15378719, 0.20378719],
                                [0.3539043, 0.2539043, 0.3039043]],
            ))

        self.assertTrue(np.isclose(nn(1, 1), ne(1, 1)).all())
        self.assertTrue(np.isclose(nn(2, 10), ne(2, 10)).all())
        self.assertTrue(np.isclose(nn(0.05, 0.1), ne(0.05, 0.1)).all())


if __name__ == '__main__':
    unittest.main()

    nn = NN(
        activation=sigmoidal,
        architecture=Architecture(
            size_input_nodes=2,
            size_output_nodes=2,
            size_hidden_nodes=2,
            output_weights=[[0.6, 0.4, 0.45], [0.6, 0.5, 0.55]],
            hidden_weights=[[0.35, 0.15, 0.2], [0.35, 0.25, 0.3]],
        ))
