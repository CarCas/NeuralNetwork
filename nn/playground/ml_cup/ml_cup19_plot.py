from nn import *
import numpy as np
from nn.utilities import read_ml_cup_tr, plot
from timeit import timeit

if __name__ == '__main__':
    ml_cup_training_dataset = read_ml_cup_tr()
    design_set, testing_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)
    training_set, validation_set = split_dataset(design_set, percentage=2/3, to_shuffle=True, seed=0)

    np.random.seed()

    seed = 5
    learning_algorithm = minibatch(0.5)
    epochs_limit = 5013
    size_hidden_layers = [75, 75]
    activation_hidden = tanh
    activation = identity
    eta = 0.0085
    alpha = 0.6
    alambd = 0
    eta_decay = 0
    patience = 10

    nn = NeuralNetwork(
        seed=seed,
        epochs_limit=epochs_limit,
        learning_algorithm=learning_algorithm,
        n_init=1,
        epsilon=0,
        error_calculator=ErrorCalculator.MEE,
        architecture=MultilayerPerceptron(
            size_hidden_layers=size_hidden_layers,
            eta=eta,
            alpha=alpha,
            alambd=alambd,
            activation=activation,
            activation_hidden=activation_hidden,
        ),
        eta_decay=eta_decay,
        patience=patience,
    )

    print(timeit(lambda: nn.fit(training_set, validation_set, testing_set), number=1))

    print('mee', nn.compute_error(training_set), nn.compute_error(validation_set), nn.compute_error(testing_set))

    training_curve = nn.training_curve
    validation_curve = nn.validation_curve
    testing_curve = nn.testing_curve

    plot(training_curve, validation=validation_curve, testing=testing_curve)
