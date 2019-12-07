from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, LearningAlgorithm
from nn import ErrorCalculator
from nn.playground.utilities import read_monk_1_tr, read_monk_1_ts, plot


if __name__ == '__main__':
    train_data = read_monk_1_tr()
    test_data = read_monk_1_ts()

    nn = NN(
        seed=4,
        activation=sigmoid,
        epochs_limit=500,
        eta=0.4,
        learning_algorithm=LearningAlgorithm.ONLINE,
        architecture=MultilayerPerceptron(6, 3, 1),
        n_init=1,
    )

    nn.train(train_data, test_data)

    nn.error_calculator = ErrorCalculator.MIS
    print(nn.compute_error(train_data), nn.compute_error(test_data))

    nn.error_calculator = ErrorCalculator.MIS
    training_error = nn.compute_learning_curve(train_data)
    testing_error = nn.compute_learning_curve(test_data)
    plot(training_error, testing_error, False)
    nn.error_calculator = ErrorCalculator.MSE
    training_error = nn.compute_learning_curve(train_data)
    testing_error = nn.compute_learning_curve(test_data)
    plot(training_error, testing_error)
