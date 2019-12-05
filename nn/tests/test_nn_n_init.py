import unittest

from nn.neural_network import ErrorTypes
from nn import NeuralNetwork as NN, sigmoid
from nn.architectures.multilayer_perceptron import MultilayerPerceptron
from nn.tests.utilities import monk1_train as train_data, monk1_test as test_data


class TestNInit(unittest.TestCase):
    def test_n_init(self):
        nn = NN(
            seed=4,
            activation=sigmoid,
            epochs_limit=5,
            architecture=MultilayerPerceptron(
                size_input_layer=6,
                size_output_layer=1,
                sizes_hidden_layers=[3],

            ),
            error_types=[ErrorTypes.MSE],
            n_init=3,
        )

        nn.train(train_data, test_data)
        self.assertEqual(nn.current_training_error[0], 0.24624873676322775)


if __name__ == '__main__':
    unittest.main()
