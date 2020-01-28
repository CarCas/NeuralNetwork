from nn import *
import numpy as np
from nn.playground.utilities import read_ml_cup_tr, plot

if __name__ == '__main__':
    seed = 0
    learning_algorithm = batch
    epochs_limit = 1000
    n_init = 1
    size_hidden_layers = (130,)
    activation = identity
    activation_hidden = tanh
    eta = 0.015
    alpha = 0.9
    alambd = 0
    eta_decay = 0
    error_calculator = ErrorCalculator.MEE
    epsilon = 0
    patience = 10

    ml_cup_training_dataset = read_ml_cup_tr()

    np.random.seed()
    train_data, test_set = split_dataset(ml_cup_training_dataset, to_shuffle=True)

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
        patience=patience,
    )

    np.random.seed()
    train_set, validation_set = split_dataset(train_data, 2/3, to_shuffle=True)

    val_result = validation(nn, train_set, validation_set, ErrorCalculator.MEE)

    print(val_result)

    nn.error_calculator = ErrorCalculator.MEE
    training_curve = nn.compute_learning_curve(train_set)
    validation_curvee = nn.compute_learning_curve(validation_set)
    plot(training_curve, validation=validation_curvee, title='mlcup19 - MEE', log=False)

    nn_good = NeuralNetwork(
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
        patience=patience,
    )

    nn_good.fit(train_data)

    nn_good.error_calculator = ErrorCalculator.MEE
    print('mee', nn_good.compute_error(train_data), nn_good.compute_error(test_set))

    nn_good.error_calculator = ErrorCalculator.MEE
    training_curve = nn_good.compute_learning_curve(train_data)
    testing_curve = nn_good.compute_learning_curve(test_set)

    print('mee, last curve value', training_curve[-1], testing_curve[-1])

    idx, score = error_calculator.choose(testing_curve)
    print(dict(epoch=idx, score_validation=score))

    plot(training_curve, testing=testing_curve, title='mlcup19 - MEE', log=True)
