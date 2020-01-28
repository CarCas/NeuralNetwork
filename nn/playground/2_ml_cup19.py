import multiprocessing
from typing import Mapping, Sequence, Any, Dict

from nn import *
from nn.playground.utilities import read_ml_cup_tr
import numpy as np

if __name__ == '__main__':

    ml_cup_training_dataset = read_ml_cup_tr()

    train_data, test_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)

    np.random.seed()

    params_nn: Dict[str, Sequence[Any]] = dict(
        error_calculator=[ErrorCalculator.MEE],
        learning_algorithm=[batch],
        epochs_limit=[100],
        n_init=[1],
        epsilon=[0],
        patience=[10],
    )

    params_architecture: Mapping[str, Sequence[Any]] = dict(
        size_hidden_layers=(list(map(lambda x: (x,), list(np.arange(100, 150, 5))))),
        activation=[identity],
        activation_hidden=[tanh],
        eta=[0.015],
        alpha=[0.9],
        alambd=[0],
        eta_decay=[0],
    )

    cv_params: Mapping[str, Any] = dict(
        cv=10,
        error_calculator=ErrorCalculator.MEE,
        to_shuffle=True,
    )

    grid_search_results = grid_search(
        train_data,
        params_nn=params_nn,
        params_architecture=params_architecture,
        cv_params=cv_params,

        n_jobs=1,
    )

    write_on_file(grid_search_results, filename='results/2_mlcup')
