from utilities import read_ml_cup_tr, read_monk_1_tr, read_monk_1_ts
import matplotlib.pyplot as plt
import time
from nn import MultilayerPerceptron, sigmoid, identity
from nn.architectures.multilayer_perceptron.neural_network import MLPMatrix

import numpy as np


def test_ml_cup19():
    data_mlcup = read_ml_cup_tr()

    data_train = read_monk_1_tr()
    data_test = read_monk_1_ts()

    data = data_mlcup

    input_size = len(data[0][0])
    output_size = len(data[0][1])

    start_time = time.time()
    nn = MLPMatrix(input_size, 100, output_size, eta=0.05, activation=identity, activation_hidden=sigmoid)
    x, d = zip(*data)
    errors = []
    for _ in range(1000):
        nn.train(data)
        out = nn(*x)
        errors.append(np.mean(np.square(np.subtract(d, out))))

    print('--- %s seconds ---' % (time.time() - start_time))
    plt.plot(errors, label='training_MSE', linestyle='-')
    plt.show()


if __name__ == '__main__':
    test_ml_cup19()
