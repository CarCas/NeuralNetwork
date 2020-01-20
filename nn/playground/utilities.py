from nn.types import Pattern
from typing import Sequence, Dict, Set, Hashable, Tuple, List
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import OneHotEncoder
import numpy as np


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
            print(k, v)
            encoder[(k, v)] = ohe

    return encoder


def transform(
    dataset: Sequence[Sequence[Hashable]],
    encoder: Encoder
) -> Sequence[Sequence[int]]:
    lst = []
    for d in dataset:
        lst2 = []
        for idx, val in enumerate(d):
            lst2.extend(encoder[idx, val])
        lst.append(np.array(lst2))
    return np.array(lst)


def encode_categorical(train_data, test_data):
    train_without_target, train_target = list(zip(*train_data))
    test_without_target, test_target = list(zip(*test_data))

    enc = OneHotEncoder()

    enc.fit(train_without_target)

    train_without_target = enc.transform(train_without_target).toarray()
    test_without_target = enc.transform(test_without_target).toarray()

    train_data_norm: Sequence[Pattern] = list(zip(train_without_target, train_target))
    test_data_norm: Sequence[Pattern] = list(zip(test_without_target, test_target))

    return train_data_norm, test_data_norm


def encode_categorical2(train_data, test_data):
    train_without_target, train_target = list(zip(*train_data))
    test_without_target, test_target = list(zip(*test_data))

    enc = one_hot_encoder(train_without_target)

    train_without_target = transform(train_without_target, enc)
    test_without_target = transform(test_without_target, enc)

    train_data_norm: Sequence[Pattern] = list(zip(train_without_target, train_target))
    test_data_norm: Sequence[Pattern] = list(zip(test_without_target, test_target))

    return train_data_norm, test_data_norm


def plot(training=[], testing=[], show=True):
    if len(training):
        plt.plot(training, label='training', linestyle='-')
    if len(testing):
        plt.plot(testing, label='testing', linestyle='--')
    if show:
        plt.legend()
        plt.show()


def read_file(path, size_target: Sequence[int] = []):
    with open(path) as f:
        data = f.readlines()
    data = [line.split(' ') for line in data]
    data = tuple(map(
        lambda el: (
            tuple(map(lambda lx: float(lx), el[2:-1])),
            [float(el[1])]),
        data))
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
                    list((map(lambda input: float(input), row[1:-2]))),
                    list((map(lambda input: float(input), row[-2:]))),
                ), readCSV))
    return _ml_cup_tr


_monk1_train = None
_monk1_test = None
_monk2_train = None
_monk2_test = None
_monk3_train = None
_monk3_test = None


def read_monk_1_tr() -> Sequence[Pattern]:
    global _monk1_train
    if _monk1_train is None:
        _monk1_train = read_monk_file('1.train')
    return _monk1_train


def read_monk_1_ts() -> Sequence[Pattern]:
    global _monk1_test
    if _monk1_test is None:
        _monk1_test = read_monk_file('1.test')
    return _monk1_test


def read_monk_2_tr() -> Sequence[Pattern]:
    global _monk2_train
    if _monk2_train is None:
        _monk2_train = read_monk_file('2.train')
    return _monk2_train


def read_monk_2_ts() -> Sequence[Pattern]:
    global _monk2_test
    if _monk2_test is None:
        _monk2_test = read_monk_file('2.test')
    return _monk2_test


def read_monk_3_tr() -> Sequence[Pattern]:
    global _monk3_train
    if _monk3_train is None:
        _monk3_train = read_monk_file('3.train')
    return _monk3_train


def read_monk_3_ts() -> Sequence[Pattern]:
    global _monk3_test
    if _monk3_test is None:
        _monk3_test = read_monk_file('3.test')
    return _monk3_test
