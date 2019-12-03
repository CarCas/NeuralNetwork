from nn.architecture import Architecture
import unittest
import numpy as np
import matplotlib.pyplot as plt

from nn import NeuralNetwork as NN, Architecture, sigmoidal, Online, Batch, ErrorTypes
from nn.tests.monk import train_data, test_data


class TestMonk(unittest.TestCase):
    def test_monk1(self):
        np.random.seed(10)

        nn = NN(
            activation_output=sigmoidal,
            architecture=Architecture(
                size_input_nodes=6,
                size_output_nodes=1,
                size_hidden_nodes=10,
            ))

        nn.train(train_data, test_data, eta=0.5, epoches=100)

        # plt.plot(nn.training_errors)
        # plt.plot(nn.testing_errors)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
