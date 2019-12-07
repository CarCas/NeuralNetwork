from typing import Sequence
import matplotlib.pyplot as plt
import csv


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


def read_monk_1_tr():
    global _monk1_train
    if _monk1_train is None:
        _monk1_train = read_monk_file('1.train')
    return _monk1_train


def read_monk_1_ts():
    global _monk1_test
    if _monk1_test is None:
        _monk1_test = read_monk_file('1.test')
    return _monk1_test


def read_monk_2_tr():
    global _monk2_train
    if _monk2_train is None:
        _monk2_train = read_monk_file('2.train')
    return _monk2_train


def read_monk_2_ts():
    global _monk2_test
    if _monk2_test is None:
        _monk2_test = read_monk_file('2.test')
    return _monk2_test


def read_monk_3_tr():
    global _monk3_train
    if _monk3_train is None:
        _monk3_train = read_monk_file('3.train')
    return _monk3_train


def read_monk_3_ts():
    global _monk3_test
    if _monk3_test is None:
        _monk3_test = read_monk_file('3.test')
    return _monk3_test