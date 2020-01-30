import multiprocessing
from typing import Mapping, Sequence, Any, Dict

from nn import *
from nn.utilities import read_ml_cup_tr
import numpy as np

if __name__ == '__main__':
    ml_cup_training_dataset = read_ml_cup_tr()
    design_set, testing_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)

    np.random.seed()

    training_set, validation_set = split_dataset(design_set, to_shuffle=True)

    params_nn: Dict[str, Sequence[Any]] = dict(
        error_calculator=[ErrorCalculator.MEE],
        learning_algorithm=[minibatch(0.05)],
        epochs_limit=[10000],
        n_init=[1],
        epsilon=[0],
        patience=[10],
    )

    params_architecture: Mapping[str, Sequence[Any]] = dict(
        size_hidden_layers=[[50, 50], [100, 100]],
        activation=[identity],
        activation_hidden=[tanh],
        eta=[0.009, 0.008],
        alpha=[0.5, 0.6],
        alambd=[0],
        eta_decay=[0],
    )

    validation_params: Mapping[str, Any] = dict(
        validation_set=validation_set,
        error_calculator=ErrorCalculator.MEE,
    )

    grid_search_results = grid_search(
        training_set,
        params_nn=params_nn,
        params_architecture=params_architecture,
        validation_params=validation_params,

        n_jobs=1,
    )

    write_on_file(grid_search_results, filename='results/example_search')
