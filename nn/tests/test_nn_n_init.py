from nn.neural_network import ErrorTypes
from nn.architecture import Architecture
import unittest
from nn import NeuralNetwork as NN, Architecture, Batch
from nn.activation_function import sigmoidal
from nn.tests.utilities import monk1_train as train_data, monk1_test as test_data


class TestNInit(unittest.TestCase):
    def test_n_init(self):
        nn = NN(
            seed=4,
            learning_algorithm=Batch(),
            activation_output=sigmoidal,
            epochs_limit=5,
            architecture=Architecture(
                size_input_nodes=6,
                size_output_nodes=1,
                size_hidden_nodes=3,
            ),
            error_types=[ErrorTypes.MSE],
            n_init=3,
        )

        nn.train(train_data, test_data)
        self.assertEqual(nn.current_training_error[0], 0.24624873676322775)


if __name__ == '__main__':
    unittest.main()
