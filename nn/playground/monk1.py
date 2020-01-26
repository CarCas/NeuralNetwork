from typing import Mapping, Sequence, Any, Dict

from nn import ErrorCalculator
from nn import sigmoid, batch
from nn.playground.utilities import read_monk
from nn.activation_functions import relu, tanh, tanh_classification
from nn.validation import grid_search, write_on_file
import multiprocessing as mp

train_data, test_data = read_monk(1)

params_nn: Dict[str, Sequence[Any]] = dict(
    error_calculator=[ErrorCalculator.MSE],
    learning_algorithm=[batch],
    epochs_limit=[500],
    n_init=[10],
    epsilon=[1e-05],
    patience=[10],
)
params_architecture: Mapping[str, Sequence[Any]] = dict(
    size_hidden_layers=[(2,)],
    activation=[sigmoid, tanh_classification],
    activation_hidden=[relu, tanh],
    eta=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    alpha=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    alambd=[0, 0.0001, 0.001],
    eta_decay=[0, 0.001],
)

cv_params: Mapping[str, Any] = dict(
    cv=10,
    error_calculator=ErrorCalculator.ACC,
    to_shuffle=True,
)

# validation_params: Mapping[str, Any] = dict(
#     validation_set=test_data,
#     error_calculator=ErrorCalculator.ACC,
#     to_shuffle=True,
# )

grid_search_results = grid_search(
    train_data,
    params_nn=params_nn,
    params_architecture=params_architecture,
    cv_params=cv_params,
    # validation_params=validation_params,

    n_jobs=7,
)

write_on_file(grid_search_results[::-1], 'monk1')
