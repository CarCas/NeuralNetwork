import unittest
from nn import NeuralNetwork as NN
from nn.activation_function import sigmoidal
import random


# monks-1: con threshold > #inputs, range_weights = (0.35, 0.60] ---> Test passed
# - ho provato anche con range_weights=0.60005 e il test viene superato, pero' credo sia dovuto al fatto che
# viene approssimato a 0.6
# - ho provato con range_weights=0.6005 e il test non viene superato, quindi magari viene approssimato a 0.601,
# e quindi fallisce perchè è maggiore di 0.6
class TestMonk(unittest.TestCase):
    def test_monk1(self):
        random.seed(3)
        nn = NN(
            activation=sigmoidal,
            architecture=NN.Architecture(
                number_inputs=6,
                number_outputs=1,
                number_hidden=5,
                threshold=8,
                range_weights=.5
            ))

        with open('../../monks/monks-2.train') as f:
            train_data = f.readlines()
        train_data = [line.split(' ') for line in train_data]
        train_data = tuple(map(
            lambda el: (
                tuple(map(lambda x: float(x), el[2:-1])),
                [float(el[1])]),
            train_data))

        with open('../../monks/monks-2.test') as f:
            test_data = f.readlines()
        test_data = [line.split(' ') for line in test_data]
        test_data = tuple(map(
            lambda el: (
                tuple(map(lambda x: float(x), el[2:-1])),
                [float(el[1])]),
            test_data))

        nn.train(train_data, 127, 0.5)

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
