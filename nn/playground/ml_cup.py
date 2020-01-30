from nn.playground.utilities import read_ml_cup_tr_normalized
import multiprocessing
from typing import Mapping, Sequence, Any, Dict

from nn import *
from nn.playground.utilities import read_ml_cup_tr
import numpy as np


ml_cup_training_dataset = read_ml_cup_tr_normalized()
train_data, test_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)
train_set, validation_set = split_dataset(train_data, percentage=2/3, to_shuffle=True, seed=0)

np.random.seed()

params_nn: Dict[str, Sequence[Any]] = dict(
    error_calculator=[ErrorCalculator.MEE],
    learning_algorithm=[minibatch(0.05)],
    epochs_limit=[500],
    n_init=[1],
    epsilon=[0],
    patience=[10],
)

params_architecture: Mapping[str, Sequence[Any]] = dict(
    size_hidden_layers=[[100], [50, 50], [50, 20], [50, 50, 50], [50, 50, 20]],
    activation=[identity],
    activation_hidden=[sigmoid, tanh, relu],
    eta=[0.1, 0.01, 0.001, 0.0001],
    alpha=[0, 0.5, 0.75, 0.9],
    alambd=[0, 0.00001, 0.0001, 0.001],
    eta_decay=[0],
)

cv_params: Mapping[str, Any] = dict(
    cv=3,
    error_calculator=ErrorCalculator.MEE,
    to_shuffle=True,
)

validation_params: Mapping[str, Any] = dict(
    validation_set=validation_set,
    error_calculator=ErrorCalculator.MEE,
)

grid_search_results = grid_search(
    train_data,
    params_nn=params_nn,
    params_architecture=params_architecture,
    # cv_params=cv_params,
    validation_params=validation_params,

    n_jobs=1,
)

write_on_file(grid_search_results, 'results/ml_cup_nocv')
