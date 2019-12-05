from unittest.util import three_way_cmp

from nn import NeuralNetwork as NN, MultilayerPerceptron, sigmoid, Online, Batch, ErrorTypes
from nn.tests.utilities import (
    monk1_train as train_data,
    monk1_test as test_data,
    plot
)


def test_monk():
    nn = NN(
        seed=4,
        activation=sigmoid,
        epochs_limit=1000,
        eta=0.99,
        architecture=MultilayerPerceptron(
            learning_algorithm=Batch(),
            size_input_layer=6,
            size_output_layer=1,
            sizes_hidden_layers=[10],
        ),
        error_types=[ErrorTypes.MSE],
        n_init=1,
        verbose=1,
    )

    nn.train(train_data, test_data)

    print('training MSE:', nn.compute_error(train_data, ErrorTypes.MSE))
    print('testing MSE:', nn.compute_error(test_data, ErrorTypes.MSE))

    print(nn.current_training_error)

    print('training MIS:', nn.compute_error(train_data, ErrorTypes.MIS))
    print('testing MIS:', nn.compute_error(test_data, ErrorTypes.MIS))

    plot(nn)


if __name__ == '__main__':
    test_monk()
