import abc
from typing import List, Optional, Sequence, Tuple, MutableSequence, Any, Mapping, NamedTuple, Dict
import numpy as np
from itertools import product
from nn.types import Architecture, ActivationFunction, Pattern
from nn.learning_algorithm import LearningAlgorithm, batch, online
from nn.error_calculator import ErrorCalculator
from nn import NeuralNetwork, MultilayerPerceptron, NNParams, MLPParams, sigmoid
from nn.playground.utilities import read_monk
from multiprocessing import Pool


class ValidationResult(NamedTuple):
    epoch: int
    score_validation: float


class KFoldCVResult(NamedTuple):
    score: float
    std: float


class GridSearchResult(NamedTuple):
    params: NNParams
    k_fold_cv_result: KFoldCVResult


def validation(
        nn: NeuralNetwork,
        training_set: Sequence[Pattern],
        validation_set: Sequence[Pattern],
        error_calculator: ErrorCalculator = ErrorCalculator.MSE
) -> ValidationResult:
    nn.fit(training_set)

    learning_curve_validation = nn.compute_learning_curve(validation_set, error_calculator)

    idx, score = error_calculator.choose(learning_curve_validation)

    return ValidationResult(
        epoch=idx,
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
) -> Tuple[Sequence[Pattern], Sequence[Pattern]]:
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


class GridSearchTaskParams(NamedTuple):
    dataset: Sequence[Pattern]
    cv_params: Mapping[str, Any]
    pm: Dict[str, Any]


def grid_search_task(params: GridSearchTaskParams) -> GridSearchResult:
    pm_with_architecture = dict(**params.pm)
    pm_with_architecture['architecture'] = MultilayerPerceptron(**params.pm['architecture'])
    nn = NeuralNetwork(**pm_with_architecture)
    kf = k_fold_CV(nn, params.dataset, **params.cv_params)
    params.pm.update(architecture=MLPParams(**params.pm['architecture']))
    typed_pm: NNParams = NNParams(**params.pm)

    # print('done')
    return GridSearchResult(typed_pm, kf)


def grid_search(
        dataset: Sequence[Pattern],
        params_nn: Mapping[str, Sequence[Any]],
        params_architecture: Mapping[str, Sequence[Any]],
        cv_params: Mapping[str, Any],

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

    print("run to generate:", len(params), "combinations")

    pool = Pool(processes=n_jobs)
    pool_params = map(lambda params: GridSearchTaskParams(*params), product([dataset], [cv_params], params))
    results: Sequence[GridSearchResult] = pool.map(grid_search_task, pool_params)

    return sorted(results, key=lambda x: x[1][0])
