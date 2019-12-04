import matplotlib.pyplot as plt


def plot(training_errors, testing_errors):
    plt.plot(training_errors, label='training', linestyle='-')
    plt.plot(testing_errors, label='testing', linestyle='--')
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
