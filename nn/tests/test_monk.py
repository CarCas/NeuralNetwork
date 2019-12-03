from nn.architecture import Architecture
import unittest
from nn import NeuralNetwork as NN, Architecture
from nn.activation_function import sigmoidal
import numpy as np


# monks-1: con threshold > #inputs, range_weights = (0.35, 0.60] ---> Test passed
# - ho provato anche con range_weights=0.60005 e il test viene superato, pero' credo sia dovuto al fatto che
# viene approssimato a 0.6
# - ho provato con range_weights=0.6005 e il test non viene superato, quindi magari viene approssimato a 0.601,
# e quindi fallisce perchè è maggiore di 0.6
class TestMonk(unittest.TestCase):
    def test_monk1(self):
        np.random.seed(3)
        nn = NN(
            activation_hidden=sigmoidal,
            early_stopping=71,
            epsilon=1e-3,
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

        # nn.fill_error_lists(train_data, test_data, 0.5, epoch_number=100)  # 127 --> passa i test gia' con 100
        # nn.fill_error_lists(train_data, test_data, 0.5)  # 127 --> passa i test gia' con 100

        nn.train(train_data, test_data, eta=0.5)
        train_errs = nn.get_training_errors()
        print("TRAINING ERRORS: ")
        for e in train_errs:
            print(e)

        # nn.test(test_data)
        test_errs = nn.get_testing_errors()
        print("TESTING ERRORS:")
        for e in test_errs:
            print(e)

        # for e in nn.training_errors:
        #    print(e)
        # while true
        # nn.train(train_data, 1, 0.5)
        # print(nn.test(train_data))

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


if __name__ == '__main__':
    unittest.main()


# TEST PASSED USING
# nn = NN(
#     activation=sigmoidal,
#     early_stopping=71,
#     epsilon=1e-2, 1e-3, 1e-4, 1e-7
#     architecture=NN.Architecture(
#         number_inputs=6,
#         number_outputs=1,
#         number_hidden=5,
#         threshold=4,
#         range_weights=.5
#     ))
# TRAIN 0.3344626053485594
# TEST 1.8598428089999413
