from nn.types import Pattern
from typing import Sequence, Dict, Set, Hashable, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import csv


Encoder = Dict[Tuple[int, Hashable], Sequence]


def one_hot_encoder(
    train_without_target: Sequence[Sequence[Hashable]]
) -> Encoder:
    values2group: List[Set[Hashable]] = [set() for _ in train_without_target[0]]
    for t in train_without_target:
        for idx, el in enumerate(t):
            values2group[idx].add(el)

    encoder: Encoder = {}
    for k, val in enumerate(values2group):
        valdim = len(val)
        for idx, v in enumerate(val):
            ohe = np.zeros(valdim)
            ohe[idx] = 1
            encoder[(k, v)] = ohe

    return encoder


def transform(
    dataset: Sequence[Sequence[Hashable]],
    encoder: Encoder
) -> Sequence[Sequence[int]]:
    lst = []
    for d in dataset:
        lst2: List[float] = []
        for idx, val in enumerate(d):
            lst2.extend(encoder[idx, val])
        lst.append(np.array(lst2))
    return np.array(lst)


def encode_categorical(train_data, test_data):
    train_without_target, train_target = list(zip(*train_data))
    test_without_target, test_target = list(zip(*test_data))

    enc = one_hot_encoder(train_without_target)

    train_without_target = transform(train_without_target, enc)
    test_without_target = transform(test_without_target, enc)

    train_data_norm: Sequence[Pattern] = list(zip(train_without_target, train_target))
    test_data_norm: Sequence[Pattern] = list(zip(test_without_target, test_target))

    return train_data_norm, test_data_norm


def plot(training=[], validation=[], testing=[], show=True, log=False, title=''):
    if log:
        plot_f = plt.loglog
    else:
        plot_f = plt.plot

    if len(training):
        plot_f(training, label='training', linestyle='-')
    if len(validation):
        plot_f(validation, label='validation', linestyle='dotted')
    if len(testing):
        plot_f(testing, label='testing', linestyle='--')
    if show:
        plt.title(title)
        plt.legend()
        plt.show()


def read_file(path, size_target: Sequence[int] = []):
    with open(path) as f:
        lines = list(csv.reader(f, delimiter=' '))

        data = tuple(map(
            lambda el: (
                tuple(map(lambda lx: float(lx), el[2:-1])),
                [float(el[1])]
            ), lines))

    return data


def read_monk_file(monk_name):
    return read_file('monks/monks-' + monk_name)


_ml_cup_tr = None


def read_ml_cup_tr():
    global _ml_cup_tr
    if _ml_cup_tr is None:
        with open('ML-CUP19/ML-CUP19-TR.csv', mode='r') as file:
            readCSV = csv.reader(file, delimiter=',')
            _ml_cup_tr = tuple(map(
                lambda row: (
                    list((map(lambda _input: float(_input), row[1:-2]))),
                    list((map(lambda _input: float(_input), row[-2:]))),
                ), readCSV))
    return _ml_cup_tr


_ml_cup_tr = None


def read_ml_cup_ts():
    global _ml_cup_tr
    if _ml_cup_tr is None:
        with open('ML-CUP19/ML-CUP19-TS.csv', mode='r') as file:
            readCSV = csv.reader(file, delimiter=',')
            _ml_cup_tr = tuple(map(
                lambda row: (
                    list((map(lambda _input: float(_input), row[1:-2]))),
                    list((map(lambda _input: float(_input), row[-2:]))),
                ), readCSV))
    return _ml_cup_tr


_monks: Dict[Hashable, Tuple[Sequence[Pattern], Sequence[Pattern]]] = {}


def read_monk(i: int) -> Tuple[Sequence[Pattern], Sequence[Pattern]]:
    global _monks
    if i not in _monks:
        _monks[i] = encode_categorical(read_monk_file('{}.train'.format(i)), read_monk_file('{}.test'.format(i)))
    return _monks[i]
