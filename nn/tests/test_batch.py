# import unittest
# import numpy as np

# from nn import NeuralNetwork as NN, Architecture, Batch
# from nn.activation_function import sigmoidal


# def sigmoidal_test(x):
#     return 1/(1 + np.exp(-x))


# def derivate_test(x):
#     return x * (1 - x)


# class TestBatch(unittest.TestCase):
#     def setUp(self):
#         self.kwargs = dict(
#             learning_algorithm=Batch(),
#             activation=sigmoidal,
#             architecture=Architecture(
#                 size_input_nodes=2,
#                 size_output_nodes=2,
#                 size_hidden_nodes=2,
#                 hidden_weights=[[0, 1.5, 2], [0, 3, 0.5]],
#                 output_weights=[[0, -1.5, 1.5], [0, -0.5, 2]]
#             ))

#     def test_out(self):
#         nn = NN(**self.kwargs)

#         nn(1, 1)
#         self.assertTrue(np.dot([1, 1, 1], [0, 1.5, 2]) == nn.hidden_layer.neurons[0].net)
#         self.assertTrue(np.dot([1, 1, 1], [0, 3, 0.5]) == nn.hidden_layer.neurons[1].net)
#         self.assertTrue(sigmoidal_test(nn.hidden_layer.neurons[0].net) == nn.hidden_layer.neurons[0].out)
#         self.assertTrue(sigmoidal_test(nn.hidden_layer.neurons[1].net) == nn.hidden_layer.neurons[1].out)

#         nn(2, 2)
#         self.assertTrue(np.dot([1, 2, 2], [0, 1.5, 2]) == nn.hidden_layer.neurons[0].net)
#         self.assertTrue(np.dot([1, 2, 2], [0, 3, 0.5]) == nn.hidden_layer.neurons[1].net)
#         self.assertTrue(sigmoidal_test(nn.hidden_layer.neurons[0].net) == nn.hidden_layer.neurons[0].out)
#         self.assertTrue(sigmoidal_test(nn.hidden_layer.neurons[1].net) == nn.hidden_layer.neurons[1].out)

#     def test_single_train(self):
#         nn = NN(**self.kwargs)

#         # nn.train(([1, 1], [0, 1]))

#         nn(1, 1)

#         delta_out_0 = (0 - nn.output_layer.neurons[0].out) * derivate_test(nn.output_layer.neurons[0].net)
#         delta_out_1 = (1 - nn.output_layer.neurons[1].out) * derivate_test(nn.output_layer.neurons[1].net)

#         delta_hid_0 = (delta_out_0 * -1.5 + delta_out_1 * -0.5) * derivate_test(nn.hidden_layer.neurons[0].net)
#         delta_hid_1 = (delta_out_0 * 1.5 + delta_out_1 * 2) * derivate_test(nn.hidden_layer.neurons[1].net)

#         delta_w_out_0_0 = 




#     def test_train(self):
#         # self.nn.train([
#         #     #inputs, outputs
#         #     ([1, 1], [0, 1]),
#         #     ([2, 2], [1, 1])],
#         #     eta=0.5)
#         pass


# if __name__ == '__main__':
#     unittest.main()
