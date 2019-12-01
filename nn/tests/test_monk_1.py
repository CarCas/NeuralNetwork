from nn.architecture import Architecture
import unittest
from nn import NeuralNetwork as NN, Architecture
from nn.activation_function import sigmoidal
import numpy as np


class TestMonk(unittest.TestCase):
    def test_monk1(self):
        np.random.seed(3)

        nn = NN(
            activation=sigmoidal,
            architecture=Architecture(
                size_input_nodes=6,
                size_output_nodes=1,
                size_hidden_nodes=5,
                range_weights=.2,
                threshold=4,
            ))

        with open('../../monks/monks-1.train') as f:
            train_data = f.readlines()
        train_data = [line.split(' ') for line in train_data]
        train_data = tuple(map(
            lambda el: (
                tuple(map(lambda lx: float(lx), el[2:-1])),
                [float(el[1])]),
            train_data))

        with open('../../monks/monks-1.test') as f:
            test_data = f.readlines()
        test_data = [line.split(' ') for line in test_data]
        test_data = tuple(map(
            lambda el: (
                tuple(map(lambda x: float(x), el[2:-1])),
                [float(el[1])]),
            test_data))

        data = train_data + test_data

        error = 1
        counter = 0
        while(error):
            counter += 1
            nn.train(train_data, [])
            error = 0
            for x, d in data:
                error += (round(nn(*x)[0]) - d[0])**2

            print(nn.test(train_data), error)

        print('counter', counter)
        self.assertEqual(error, 0)


if __name__ == '__main__':
    unittest.main()
