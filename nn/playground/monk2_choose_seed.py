from typing import Mapping, Sequence, Any, Dict

from nn import ErrorCalculator
from nn import sigmoid, batch
from nn.playground.utilities import read_monk
from nn.activation_functions import relu, tanh, tanh_classification
from nn.validation import grid_search, write_on_file
import multiprocessing as mp
from nn import split_dataset

train_data, test_data = read_monk(1)

train_set, validation_set = split_dataset(train_data, 2/3, to_shuffle=True, seed=0)

params_nn: Dict[str, Sequence[Any]] = dict(
    seed=list(range(100)),
    error_calculator=[ErrorCalculator.MSE],
    learning_algorithm=[batch],
    epochs_limit=[1000],
    n_init=[1],
    epsilon=[1e-05],
    patience=[10],
)
params_architecture: Mapping[str, Sequence[Any]] = dict(
    size_hidden_layers=[(2,)],
    activation=[tanh_classification],
    activation_hidden=[relu],
    eta=[0.1],
    alpha=[0.6],
    alambd=[0],
    eta_decay=[0],
)

validation_params: Mapping[str, Any] = dict(
    validation_set=validation_set,
    error_calculator=ErrorCalculator.MSE,
)

grid_search_results = grid_search(
    train_set,
    params_nn=params_nn,
    params_architecture=params_architecture,
    validation_params=validation_params,

    n_jobs=8,
)

write_on_file(grid_search_results, 'results/monk2_choose_seed')
