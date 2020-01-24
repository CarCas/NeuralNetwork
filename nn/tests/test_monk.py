import unittest

from nn import NeuralNetwork as NN, sigmoid, MultilayerPerceptron, relu
from nn.learning_algorithm import online, batch
from nn.error_calculator import ErrorCalculator
from nn.playground.utilities import read_monk


class TestMonk(unittest.TestCase):
    def test_monk1(self):
        nn = NN(
            seed=4,
            epochs_limit=400,
            learning_algorithm=batch,
            error_calculator=ErrorCalculator.MSE,
            architecture=MultilayerPerceptron(
                4,
                activation=sigmoid,
                activation_hidden=relu,
                eta=0.5,
                alambd=0,
                alpha=0.8,
            )
        )

        train_data, test_data = read_monk(1)

        nn.fit(train_data)
        train_errs = nn.compute_learning_curve(train_data, ErrorCalculator.MIS)

        test_errs = nn.compute_learning_curve(test_data, ErrorCalculator.MIS)

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
