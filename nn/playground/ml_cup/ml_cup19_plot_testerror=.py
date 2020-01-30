from nn import *
import numpy as np
from nn.utilities import read_ml_cup_tr, plot


if __name__ == '__main__':
    ml_cup_training_dataset = read_ml_cup_tr()
    train_data, test_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)

    seed = 0
    learning_algorithm = batch
    epochs_limit = 10000
    n_init = 1
    size_hidden_layers = [40, 10, 30]
    activation = identity
    activation_hidden = tanh
    eta = 0.007
    alpha = 0.7
    alambd = 1e-05
    eta_decay = 0
    error_calculator = ErrorCalculator.MEE
    epsilon = 0

    nn = NeuralNetwork(
        seed=seed,
        learning_algorithm=learning_algorithm,
        n_init=n_init,
        epochs_limit=epochs_limit,
        architecture=MultilayerPerceptron(
            size_hidden_layers=size_hidden_layers,
            activation=activation,
            activation_hidden=activation_hidden,
            eta=eta,
            alpha=alpha,
            alambd=alambd,
            eta_decay=eta_decay,
        ),
        error_calculator=error_calculator,
        epsilon=epsilon,
    )

    train_set, validation_set = split_dataset(train_data, percentage=2/3, to_shuffle=True)

    val_result = validation(nn, train_set, validation_set, ErrorCalculator.MEE)

    print('trained', len(nn.validation_curve), 'epoches')

    print(val_result)

    # nn.error_calculator = ErrorCalculator.MEE
    # validation_curvee = nn.validation_curve
    # plot(validation=validation_curvee, title='mlcup19 - MEE', log=True)

    nn_good = NeuralNetwork(
        seed=seed,
        learning_algorithm=learning_algorithm,
        n_init=n_init,
        epochs_limit=val_result.epoch,
        architecture=MultilayerPerceptron(
            size_hidden_layers=size_hidden_layers,
            activation=activation,
            activation_hidden=activation_hidden,
            eta=eta,
            alpha=alpha,
            alambd=alambd,
            eta_decay=eta_decay,
        ),
        error_calculator=error_calculator,
        epsilon=epsilon,
    )

    nn_good.fit(train_set, test_set)

    training_curve = nn_good.training_curve
    testing_curve = nn_good.validation_curve

    idx, score = nn.error_calculator.choose(testing_curve)

    nn_good.error_calculator = ErrorCalculator.MEE
    print('mee', training_curve[-1], testing_curve[-1])
    print('best accuracy:', idx+1)

    nn_good.error_calculator = ErrorCalculator.MEE
    # training_curve = nn_good.compute_learning_curve(train_data)
    # testing_curve = nn_good.compute_learning_curve(test_set)
    plot(training_curve, testing=testing_curve, title='mlcup19 - MEE')
