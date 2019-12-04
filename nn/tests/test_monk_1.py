from nn.architecture import Architecture
import unittest
import numpy as np
import matplotlib.pyplot as plt

from nn import NeuralNetwork as NN, Architecture, sigmoidal, Online, Batch, ErrorTypes
from nn.tests.monk_1 import train_data, test_data


class TestMonk(unittest.TestCase):
    def test_monk1(self):
        np.random.seed(10)

        nn = NN(
            learning_algorithm=Batch(),
            activation_output=sigmoidal,
            eta=0.3,
            max_epochs=30,
            architecture=Architecture(
                size_input_nodes=6,
                size_output_nodes=1,
                size_hidden_nodes=4
            ))

        nn().train(train_data, test_data)

        print('training', 1-nn.compute_error(train_data, ErrorTypes.MIS)/len(train_data))
        print('testing', 1-nn.compute_error(test_data, ErrorTypes.MIS)/len(test_data))

        plt.plot(nn.training_errors)
        plt.plot(nn.testing_errors)
        plt.show()


if __name__ == '__main__':
    unittest.main()
