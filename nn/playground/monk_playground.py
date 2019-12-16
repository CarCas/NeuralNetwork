from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, online, minibatch, batch
from nn import ErrorCalculator
from nn.playground.utilities import read_monk_1_tr, read_monk_1_ts, plot, read_monk_2_tr, read_monk_2_ts


if __name__ == '__main__':
    train_data = read_monk_2_tr()
    test_data = read_monk_2_ts()

    nn = NN(
        seed=4,
        activation=sigmoid,
        epochs_limit=100,
        eta=0.5,
        alpha=0.1,
        learning_algorithm=batch,
        architecture=MultilayerPerceptron(6, 2, 1),
        n_init=1,
    )

    nn.train(train_data, test_data)

    # nn.error_calculator = ErrorCalculator.MIS
    # print(nn.compute_error(train_data), nn.compute_error(test_data))

    # nn.error_calculator = ErrorCalculator.MIS
    # training_error = nn.compute_learning_curve(train_data)
    # testing_error = nn.compute_learning_curve(test_data)
    # plot(training_error, testing_error, False)
    nn.error_calculator = ErrorCalculator.MSE
    training_error = nn.compute_learning_curve(train_data)
    testing_error = nn.compute_learning_curve(test_data)
    plot(training_error, testing_error)
    nn.error_calculator = ErrorCalculator.MSE
    print(nn.compute_error(train_data), nn.compute_error(test_data))
