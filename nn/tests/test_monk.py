# import unittest
# from nn import NeuralNetwork as NN
# from nn.activation_function import sigmoidal_1 as sigmoidal


# class TestMonk(unittest.TestCase):
#     def test_monk1(self):
#         nn = NN(
#             activation=sigmoidal,
#             architecture=NN.Architecture(
#                 number_inputs=6,
#                 number_outputs=1,
#                 number_hidden=10))

#         with open('monks/monks-1.train') as f:
#             train_data = f.readlines()
#         train_data = [line.split(' ') for line in train_data]
#         train_data = tuple(map(
#             lambda el: (
#                 tuple(map(lambda x: float(x), el[2:-1])),
#                 [float(el[1])]),
#             train_data))

#         with open('monks/monks-1.test') as f:
#             test_data = f.readlines()
#         test_data = [line.split(' ') for line in test_data]
#         test_data = tuple(map(
#             lambda el: (
#                 tuple(map(lambda x: float(x), el[2:-1])),
#                 [float(el[1])]),
#             test_data))

#         try:
#             while(nn.test(train_data) > 1):
#                 nn.train(train_data, 1, 0.5)
#                 print('Error on train data:', nn.test(train_data))
#         except KeyboardInterrupt:
#             pass

#         error_train = 0
#         for x, d in train_data:
#             error_train += (round(nn(*x)[0]) - d[0])**2

#         error_test = 0
#         for x, d in test_data:
#             error_test += (round(nn(*x)[0]) - d[0])**2

#         print('train:',
#               str(((len(train_data)-error_train)/len(train_data))*100) + '%')
#         print('test: ',
#               str(((len(test_data)-error_test)/len(test_data))*100) + '%')

#         self.assertEqual(((len(train_data)-error_train)/len(train_data)), 1)
#         self.assertEqual(((len(test_data)-error_test)/len(test_data)), 1)


# if __name__ == '__main__':
#     unittest.main()
