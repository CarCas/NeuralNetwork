from nn.architectures.multilayer_perceptron.learning_algorithms import LeariningAlgorthm
from nn.neural_network import ErrorTypes
import unittest

from nn import NeuralNetwork as NN, MultilayerPerceptron, Online, sigmoid
from nn.tests.utilities import monk1_train as train_data, monk1_test as test_data


class TestMonk(unittest.TestCase):
    def test_monk1(self):
        nn = NN(
            seed=3,
            activation=sigmoid,
            epochs_limit=71,
            epsilon=1e-3,
            architecture=MultilayerPerceptron(
                learining_algorthm=Online(),
                size_input_nodes=6,
                size_output_nodes=1,
                size_hidden_nodes=5,
                range_weights=.2,
                threshold=4,
            ))

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
            error_train += (round(nn(*x)[0]) - d[0])**2

        error_test = 0
        for x, d in test_data:
            error_test += (round(nn(*x)[0]) - d[0])**2

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
