from unittest.util import three_way_cmp
from nn.architecture import Architecture
import unittest
import numpy as np

from nn import NeuralNetwork as NN, Architecture, sigmoidal, Online, Batch, ErrorTypes
from nn.tests.utilities import (
    monk1_train as train_data,
    monk1_test as test_data,
    plot
)


class TestMonk(unittest.TestCase):
    def test_monk1(self):
        nn = NN(
            seed=3,
            activation_output=sigmoidal,
            epochs_limit=72,
            architecture=Architecture(
                size_input_nodes=6,
                size_output_nodes=1,
                size_hidden_nodes=5,
                range_weights=.2,
                threshold=4
            ),
            error_types=[ErrorTypes.MSE, ErrorTypes.MIS, ErrorTypes.ACC],
            verbose=1,
        )

        nn.train(train_data, test_data)

        # plot(nn)

        self.assertEqual(nn.compute_error(train_data, ErrorTypes.MIS), 0)
        self.assertEqual(nn.compute_error(test_data, ErrorTypes.MIS), 0)


if __name__ == '__main__':
    unittest.main()
