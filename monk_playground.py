from nn import NeuralNetwork as NN, sigmoid, ErrorTypes, MultilayerPerceptron, LearningAlgorithm
from utilities import read_monk_1_tr, read_monk_1_ts, plot


def test_monk():
    nn = NN(
        seed=4,
        activation=sigmoid,
        epochs_limit=100,
        eta=0.05,
        learning_algorithm=LearningAlgorithm.ONLINE,
        architecture=MultilayerPerceptron(6, 3, 1),
        error_types=[ErrorTypes.MSE],
        n_init=1,
        verbose=2,
    )

    train_data = read_monk_1_tr()
    test_data = read_monk_1_ts()

    nn.train(train_data, test_data)

    # print('training MSE:', nn.compute_error(train_data, ErrorTypes.MSE))
    # print('testing MSE:', nn.compute_error(test_data, ErrorTypes.MSE))

    # print('training MEE:', nn.compute_error(train_data, ErrorTypes.MEE))
    # print('testing MEE:', nn.compute_error(test_data, ErrorTypes.MEE))

    # print('training MIS:', nn.compute_error(train_data, ErrorTypes.MIS))
    # print('testing MIS:', nn.compute_error(test_data, ErrorTypes.MIS))

    plot(nn)


if __name__ == '__main__':
    test_monk()
