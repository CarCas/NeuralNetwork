import multiprocessing
from typing import Mapping, Sequence, Any, Dict

from nn import *
from nn.playground.utilities import read_ml_cup_tr

if __name__ == '__main__':

    ml_cup_training_dataset = read_ml_cup_tr()

    train_set, test_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)

    params_nn: Dict[str, Sequence[Any]] = dict(
        error_calculator=[ErrorCalculator.MEE],
        learning_algorithm=[minibatch(0.1)],
        epochs_limit=[500],
        n_init=[3],
        epsilon=[0.00001],
        patience=[10],
    )

    params_architecture: Mapping[str, Sequence[Any]] = dict(
        size_hidden_layers=((10,), (20,), (50,)),
        activation=[identity],
        activation_hidden=[relu, sigmoid, tanh],
        eta=[0.01, 0.005, 0.001],
        alpha=[0, 0.3, 0.5, 0.8],
        alambd=[0, 0.0001, 0.001, 0.01],
        eta_decay=[0, 0.001, 0.01, 0.1],
    )

    cv_params: Mapping[str, Any] = dict(
        cv=3,
        error_calculator=ErrorCalculator.MEE,
        to_shuffle=True,
    )

    grid_search_results = grid_search(
        train_set,
        params_nn=params_nn,
        params_architecture=params_architecture,
        cv_params=cv_params,

        n_jobs=multiprocessing.cpu_count(),
    )

    write_on_file(grid_search_results, filename='mlcup1')
