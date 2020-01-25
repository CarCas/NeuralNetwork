from typing import Mapping, Sequence, Any, Dict

from nn import ErrorCalculator
from nn import sigmoid, batch, relu
from nn.playground.utilities import read_ml_cup_tr, read_ml_cup_ts
from nn.validation import grid_search, write_on_file

if __name__ == '__main__':
    train_data = read_ml_cup_tr()
    test_data = read_ml_cup_ts()

    params_nn: Dict[str, Sequence[Any]] = dict(
        error_calculator=[ErrorCalculator.MSE],
        learning_algorithm=[batch],
        epochs_limit=[1000],
        n_init=[10],
        epsilon=[0.00001],
        patience=[100],
    )
    params_architecture: Mapping[str, Sequence[Any]] = dict(
        size_hidden_layers=[(2,), (2, 4)],
        activation=[sigmoid, relu],
        activation_hidden=[relu],
        eta=[0.1, 0.4, 0.8],
        alpha=[0.1, 0.4, 0.8],
        alambd=[0, 0.1, 0.4],
        eta_decay=[0, 0.01],
    )

    cv_params: Mapping[str, Any] = dict(
        cv=5,
        error_calculator=ErrorCalculator.ACC,
        to_shuffle=True,
    )

    validation_params: Mapping[str, Any] = dict(
        validation_set=test_data,
        error_calculator=ErrorCalculator.ACC,
    )

    grid_search_results = grid_search(
        train_data,
        params_nn=params_nn,
        params_architecture=params_architecture,
        # cv_params=cv_params,
        validation_params=validation_params,

        n_jobs=8,
        seed=1,
    )

    for i, entry in enumerate(grid_search_results[::-1][:3]):
        for key, value in entry._asdict().items():
            print('{}:'.format(key), value)

    write_on_file(grid_search_results, filename="mlcup")
