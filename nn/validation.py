import csv
from typing import Optional, Sequence, Tuple, MutableSequence, Any, Mapping, NamedTuple, Dict
import numpy as np
from itertools import product

from nn.learning_algorithm import LearningAlgorithm
from nn.types import Pattern
from nn.error_calculator import ErrorCalculator
from nn import NeuralNetwork, MultilayerPerceptron, MLPParams
from multiprocessing import Pool
from tqdm import tqdm


class ValidationResult(NamedTuple):
    epoch: int
    score_validation: float


class KFoldCVResult(NamedTuple):
    score: float
    std: float


class NNParams(NamedTuple):
    architecture: MLPParams
    error_calculator: ErrorCalculator
    learning_algorithm: LearningAlgorithm
    epochs_limit: int
    epsilon: float
    patience: int
    n_init: int


class GridSearchResult(NamedTuple):
    params: NNParams
    score: float
    std: float


def validation(
        nn: NeuralNetwork,
        training_set: Sequence[Pattern],
        validation_set: Sequence[Pattern],
        error_calculator: ErrorCalculator = ErrorCalculator.MSE
) -> ValidationResult:
    old_error = nn.error_calculator

    nn.error_calculator = error_calculator

    nn.fit(training_set, validation_set, training_curve=False)
    learning_curve_validation = nn.validation_curve

    idx, score = error_calculator.choose(learning_curve_validation)

    nn.error_calculator = old_error

    return ValidationResult(
        epoch=idx + 1,
        score_validation=score,
    )


def shuffle(patterns: Sequence[Pattern], seed: Optional[int] = None) -> Sequence[Pattern]:
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(patterns)


def _compute_size_given_percentage(dataset: Sequence[Pattern], percentage: float):
    assert (0 < percentage < 1)
    return int(np.round(len(dataset) * percentage))


# return a tuple containing two dataset, the first one contains percentage elements
def split_dataset(
        dataset: Sequence[Pattern],
        percentage: float = 0.7,
        size: Optional[int] = None,

        to_shuffle: bool = False,
        seed: Optional[int] = None,
) -> Tuple[Sequence[Pattern], Sequence[Pattern]]:
    if to_shuffle:
        dataset = shuffle(dataset, seed)

    if size is not None:
        len_training = size
    else:
        len_training = _compute_size_given_percentage(dataset, percentage)

    return dataset[:len_training], dataset[len_training:]


def k_fold_CV(
        nn: NeuralNetwork,
        dataset: Sequence[Pattern],
        cv: int = 5,
        error_calculator: ErrorCalculator = ErrorCalculator.MSE,
        to_shuffle: bool = False,

        seed: Optional[int] = None,
) -> KFoldCVResult:
    if to_shuffle:
        dataset = shuffle(dataset, seed)

    len_training = int(np.round(len(dataset) * (cv - 1) / cv))
    shift_size = int(np.round(len_training))

    scores: MutableSequence[ValidationResult] = []

    for i in range(cv):
        training_set, validation_set = split_dataset(np.roll(dataset, shift_size * i), size=len_training)
        scores.append(validation(nn.set(), training_set, validation_set, error_calculator=error_calculator))

    scores_1 = list(map(lambda x: x[1], scores))

    score = float(np.mean(scores_1))
    std = float(np.std(scores_1))

    return KFoldCVResult(
        score=score,
        std=std,
    )


def grid_search_task_init(
        dataset_: Sequence[Pattern],
        cv_params_: Mapping[str, Any],
        validation_params_: Optional[Mapping[str, Any]]
) -> None:
    global dataset_process, cv_params_process, validation_params_process

    dataset_process, cv_params_process, validation_params_process = dataset_, cv_params_, validation_params_  # type: ignore


def grid_search_task(pm: Dict[str, Any]) -> GridSearchResult:
    pm_with_architecture = dict(**pm)
    pm_with_architecture['architecture'] = MultilayerPerceptron(**pm['architecture'])
    nn = NeuralNetwork(**pm_with_architecture)
    if validation_params_process is None:  # type: ignore
        kf = k_fold_CV(nn, dataset_process, **cv_params_process)  # type: ignore
    else:
        kf = KFoldCVResult(validation(
            nn,
            dataset_process,  # type: ignore
            **validation_params_process,  # type: ignore
        ).score_validation, 0)
    pm.update(architecture=MLPParams(**pm['architecture']))
    typed_pm: NNParams = NNParams(**pm)

    # print('done')
    return GridSearchResult(typed_pm, *kf)


def grid_search(
        dataset: Sequence[Pattern],
        params_nn: Mapping[str, Sequence[Any]],
        params_architecture: Mapping[str, Sequence[Any]],
        cv_params: Mapping[str, Any] = {},

        validation_params: Optional[Mapping[str, Any]] = None,

        n_jobs: int = 4,

        seed: Optional[int] = None,
) -> Sequence[GridSearchResult]:
    if seed is not None:
        np.random.seed(seed)

    arch_combs = list(product(*params_architecture.values()))
    nn_combs = list(product(*params_nn.values()))

    combos = list(product(arch_combs, nn_combs))
    params: MutableSequence[Dict[str, Any]] = []

    for comb in combos:
        pm_nn, pm_arch = {}, {}

        for i, key in enumerate(list(params_nn.keys())):
            pm_nn[key] = comb[1][i]
        for i, key in enumerate(list(params_architecture.keys())):
            pm_arch[key] = comb[0][i]

        pm_nn['architecture'] = pm_arch

        params.append(pm_nn)

    n_combination = len(params)

    print("run to generate:", n_combination, "combinations")
    pbar = tqdm(total=n_combination)

    pool = Pool(processes=n_jobs, initializer=grid_search_task_init, initargs=(dataset, cv_params, validation_params))

    # results: Sequence[GridSearchResult] = pool.map(grid_search_task, params)

    res = [pool.apply_async(grid_search_task, args=(params_,),
           callback=lambda _: pbar.update(1)) for params_ in params]

    results: Sequence[GridSearchResult] = [p.get() for p in res]

    return sorted(results, key=lambda x: x[1])


def write_on_file(results: Sequence[GridSearchResult], filename: str) -> None:
    nn_keys = list(results[0].params._asdict().keys())[1:]
    nn_keys = list(map(lambda x: 'nn_' + x, nn_keys))

    arch_keys = list(results[0].params.architecture._asdict().keys())
    arch_keys = list(map(lambda x: 'arch_' + x, arch_keys))

    val_keys = list(results[0]._asdict().keys())[1:]
    keys = val_keys + nn_keys + arch_keys

    values = []
    for res in results:
        nn_vals = list(res.params._asdict().values())[1:]
        arch_vals = list(res.params.architecture._asdict().values())
        val_vals = list(res[1:])
        values.append(val_vals + nn_vals + arch_vals)

    with open(filename + ".csv", "w") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows(values)
