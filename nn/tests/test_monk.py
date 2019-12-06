from nn.neural_network import ErrorTypes
import unittest

from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron
from nn.types import LearningAlgorithm
from utilities import read_monk_1_tr, read_monk_1_ts


class TestMonk(unittest.TestCase):
    def test_monk1(self):
        nn = NN(
            seed=4,
            activation=sigmoid,
            epochs_limit=80,
            eta=0.65,
            learning_algorithm=LearningAlgorithm.ONLINE,
            error_types=[ErrorTypes.MIS],
            architecture=MultilayerPerceptron(6, 4, 1)
        )

        train_data = read_monk_1_tr()
        test_data = read_monk_1_ts()

        nn.train(train_data, test_data)
        train_errs = nn.get_training_errors()
        print("TRAINING ERRORS: ")
        for e in train_errs:
            print(e)

        test_errs = nn.get_testing_errors()
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

        self.assertEqual(nn.compute_error(train_data, ErrorTypes.MIS), 0)
        self.assertEqual(nn.compute_error(test_data, ErrorTypes.MIS), 0)


if __name__ == '__main__':
    unittest.main()
