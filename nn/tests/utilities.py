from nn.neural_network import NeuralNetwork
import matplotlib.pyplot as plt


def plot(nn: NeuralNetwork):
    for error_type in nn.error_types:
        plt.plot(nn.training_errors[error_type], label='training_'+error_type.name, linestyle='-')
        plt.plot(nn.testing_errors[error_type], label='testing_'+error_type.name, linestyle='--')
    plt.legend()
    plt.show()


def read_file(path):
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
    return read_file('../../monks/monks-' + monk_name)


monk1_train = read_monk_file('1.train')
monk1_test = read_monk_file('1.test')
monk2_train = read_monk_file('2.train')
monk2_test = read_monk_file('2.test')
monk3_train = read_monk_file('3.train')
monk3_test = read_monk_file('3.test')
