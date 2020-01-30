from nn import *
from typing import *
import numpy as np
from nn.utilities import read_ml_cup_tr, plot


ml_cup_training_dataset = read_ml_cup_tr()
train_data, test_set = split_dataset(ml_cup_training_dataset, to_shuffle=True, seed=0)
train_set, validation_set = split_dataset(train_data, percentage=2/3, to_shuffle=True, seed=0)

epsilon = 0

params_nn: Dict[str, Sequence[Any]] = dict(
    seed=list(range(0, 8)),
    error_calculator=[ErrorCalculator.MEE],
    learning_algorithm=[batch],
    epochs_limit=[13000],
    n_init=[1],
    epsilon=[0],
    patience=[10],
)
params_architecture: Mapping[str, Sequence[Any]] = dict(
    size_hidden_layers=[[40, 30]],
    activation=[identity],
    activation_hidden=[tanh],
    eta=[0.007],
    alpha=[0.55],
    alambd=[1e-05],
    eta_decay=[0],
)

validation_params: Mapping[str, Any] = dict(
    validation_set=validation_set,
    error_calculator=ErrorCalculator.MEE,
)

grid_search_results = grid_search(
    train_set,
    params_nn=params_nn,
    params_architecture=params_architecture,
    validation_params=validation_params,

    n_jobs=8,
)

write_on_file(grid_search_results, 'results/ml_cup_choose_seed_0-7')
