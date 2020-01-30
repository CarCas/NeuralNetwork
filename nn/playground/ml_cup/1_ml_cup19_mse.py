import multiprocessing
from typing import Mapping, Sequence, Any, Dict

from nn import *
from nn.utilities import read_ml_cup_tr
import numpy as np

if __name__ == '__main__':

    ml_cup_training_dataset = read_ml_cup_tr()

    train_data, test_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)

    np.random.seed()

    params_nn: Dict[str, Sequence[Any]] = dict(
        error_calculator=[ErrorCalculator.MSE],
        learning_algorithm=[minibatch(0.1)],
        epochs_limit=[100],
        n_init=[3],
        epsilon=[0],
        patience=[10],
    )

    params_architecture: Mapping[str, Sequence[Any]] = dict(
        size_hidden_layers=((50,), (100,), (150,), (200,)),
        activation=[identity],
        activation_hidden=[tanh, relu],
        eta=[0.015, 0.01, 0.005, 0.001],
        alpha=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        alambd=[0, 0.001, 0.0001],
        eta_decay=[0],
    )

    cv_params: Mapping[str, Any] = dict(
        cv=10,
        error_calculator=ErrorCalculator.MSE,
        to_shuffle=True,
    )

    grid_search_results = grid_search(
        train_data,
        params_nn=params_nn,
        params_architecture=params_architecture,
        cv_params=cv_params,

        n_jobs=1,
    )

    write_on_file(grid_search_results, filename='results/1_mlcup-mse')
