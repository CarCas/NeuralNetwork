import unittest

from nn.neural_network import ErrorTypes
from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron
from utilities import read_monk_1_tr as train_data, read_monk_1_ts as test_data


class TestNInit(unittest.TestCase):
    def test_n_init(self):
        nn = NN(
            seed=4,
            activation=sigmoid,
            epochs_limit=5,
            architecture=MultilayerPerceptron(6, 3, 1),
            error_types=[ErrorTypes.MSE],
            n_init=3,
        )

        nn.train(train_data(), test_data())
        self.assertEqual(nn.current_training_error, 0.24624873676322775)


if __name__ == '__main__':
    unittest.main()
