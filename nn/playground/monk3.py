from typing import Mapping, Sequence, Any, Dict

from nn import ErrorCalculator
from nn import sigmoid, batch
from nn.playground.utilities import read_monk
from nn.activation_functions import relu, tanh, tanh_classification
from nn.validation import grid_search, write_on_file
import multiprocessing as mp

train_data, test_data = read_monk(3)

params_nn: Dict[str, Sequence[Any]] = dict(
    error_calculator=[ErrorCalculator.MSE],
    learning_algorithm=[batch],
    epochs_limit=[500],
    n_init=[5],
    epsilon=[1e-05],
    patience=[10],
)
params_architecture: Mapping[str, Sequence[Any]] = dict(
    size_hidden_layers=[(2,), (4,), (2, 2)],
    activation=[tanh_classification],
    activation_hidden=[relu],
    eta=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    alpha=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    alambd=[0, 0.001, 0.002],
    eta_decay=[0, 0.9, 0.5, 0.1],
)

cv_params: Mapping[str, Any] = dict(
    cv=3,
    error_calculator=ErrorCalculator.MSE,
    to_shuffle=True,
)

grid_search_results = grid_search(
    train_data,
    params_nn=params_nn,
    params_architecture=params_architecture,
    cv_params=cv_params,

    n_jobs=4,
)

write_on_file(grid_search_results, 'results/monk3-mse-fix')
