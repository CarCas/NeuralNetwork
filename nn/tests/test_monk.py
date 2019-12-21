import unittest

from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron
from nn.learning_algorithm import online
from nn.error_calculator import ErrorCalculator
from nn.playground.utilities import read_monk_1_tr, read_monk_1_ts


class TestMonk(unittest.TestCase):
    def test_monk1(self):
        nn = NN(
            seed=4,
            epochs_limit=80,
            learning_algorithm=online,
            error_calculator=ErrorCalculator.MIS,
            architecture=MultilayerPerceptron(6, 4, 1, activation=sigmoid, eta=0.65)
        )

        train_data = read_monk_1_tr()
        test_data = read_monk_1_ts()

        nn.train(train_data, test_data)
        train_errs = nn.compute_learning_curve(train_data)
        print("TRAINING ERRORS: ")
        for e in train_errs:
            print(e)

        test_errs = nn.compute_learning_curve(test_data)
        print("TESTING ERRORS:")
        for e in test_errs:
            print(e)

        error_train = 0
        for x, d in train_data:
            error_train += (round(nn(x)[0][-1]) - d[0])**2

        error_test = 0
        for x, d in test_data:
            error_test += (round(nn(x)[0][-1]) - d[0])**2

        print('train:',
              str(((len(train_data)-error_train)/len(train_data))*100) + '%')
        print('test: ',
              str(((len(test_data)-error_test)/len(test_data))*100) + '%')

        self.assertEqual(error_train, 0)
        self.assertEqual(error_test, 0)

        nn.error_calculator = ErrorCalculator.MIS
        self.assertEqual(nn.compute_error(train_data), 0)
        self.assertEqual(nn.compute_error(test_data), 0)


if __name__ == '__main__':
    unittest.main()
