from unittest.util import three_way_cmp
from nn.architecture import Architecture

from nn import NeuralNetwork as NN, Architecture, sigmoidal, Online, Batch, ErrorTypes
from nn.tests.utilities import (
    monk1_train as train_data,
    monk1_test as test_data,
    plot
)


def test_monk():
    nn = NN(
        seed=4,
        learning_algorithm=Batch(),
        activation_output=sigmoidal,
        epochs_limit=20,
        architecture=Architecture(
            size_input_nodes=6,
            size_output_nodes=1,
            size_hidden_nodes=10,
        ),
        error_types=[ErrorTypes.MSE],
        n_init=3,
        verbose=1,
    )

    nn.train(train_data, test_data)
    print(0, 'train', nn.compute_error(train_data, ErrorTypes.MSE))
    print(0, 'test', nn.compute_error(test_data, ErrorTypes.MSE))

    plot(nn)

    # nn()
    # nn.train(train_data, test_data)
    # print(1, 'train', nn.compute_error(train_data, ErrorTypes.MSE))
    # print(1, 'test', nn.compute_error(test_data, ErrorTypes.MSE))

    # self.assertEqual(nn.compute_error(train_data, ErrorTypes.MIS), 0)
    # self.assertEqual(nn.compute_error(test_data, ErrorTypes.MIS), 0)


if __name__ == '__main__':
    test_monk()
