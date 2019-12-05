import unittest
import numpy as np

from nn import MultilayerPerceptron, NeuralNetwork as NN
from nn.activation_function import sigmoid


def sigmoid_test(x):
    return 1/(1 + np.exp(-x))


def derivate_test(x):
    return x.out * (1-x.out)


class TestBatch(unittest.TestCase):
    # def setUp(self):
    #     self.architecture = MultilayerPerceptron(
    #             size_input_layer=2,
    #             size_output_layer=2,
    #             sizes_hidden_layers=[2],
    #             hidden_weights=[[0, 1.5, 2], [0, 3, 0.5]],
    #             output_weights=[[0, -1.5, 1.5], [0, -0.5, 2]]
    #     )

    #     self.p0_delta_w_out_00 = -1
    #     self.p1_delta_w_out_00 = -1
    #     self.p0_delta_w_out_01 = -1
    #     self.p1_delta_w_out_01 = -1
    #     self.p0_delta_w_out_02 = -1
    #     self.p1_delta_w_out_02 = -1
    #     self.p0_delta_w_out_10 = -1
    #     self.p1_delta_w_out_10 = -1
    #     self.p0_delta_w_out_11 = -1
    #     self.p1_delta_w_out_11 = -1
    #     self.p0_delta_w_out_12 = -1
    #     self.p1_delta_w_out_12 = -1
    #     self.p0_delta_w_hid_00 = -1
    #     self.p1_delta_w_hid_00 = -1
    #     self.p0_delta_w_hid_01 = -1
    #     self.p1_delta_w_hid_01 = -1
    #     self.p0_delta_w_hid_02 = -1
    #     self.p1_delta_w_hid_02 = -1
    #     self.p0_delta_w_hid_10 = -1
    #     self.p1_delta_w_hid_10 = -1
    #     self.p0_delta_w_hid_11 = -1
    #     self.p1_delta_w_hid_11 = -1
    #     self.p0_delta_w_hid_12 = -1
    #     self.p1_delta_w_hid_12 = -1

    # def test_out(self):
    #     nn = self.architecture(sigmoid)

    #     nn(1, 1)
    #     self.assertTrue(np.dot([1, 1, 1], [0, 1.5, 2]) == nn.hidden_layers[0].neurons[0].net)
    #     self.assertTrue(np.dot([1, 1, 1], [0, 3, 0.5]) == nn.hidden_layers[0].neurons[1].net)
    #     self.assertTrue(sigmoid_test(nn.hidden_layers[0].neurons[0].net) == nn.hidden_layers[0].neurons[0].out)
    #     self.assertTrue(sigmoid_test(nn.hidden_layers[0].neurons[1].net) == nn.hidden_layers[0].neurons[1].out)
    #     np.isclose(derivate_test(nn.hidden_layers[0].neurons[0]), nn.hidden_layers[0].neurons[0].fprime)
    #     np.isclose(derivate_test(nn.hidden_layers[0].neurons[1]), nn.hidden_layers[0].neurons[1].fprime)

    #     nn(2, 2)
    #     self.assertTrue(np.dot([1, 2, 2], [0, 1.5, 2]) == nn.hidden_layers[0].neurons[0].net)
    #     self.assertTrue(np.dot([1, 2, 2], [0, 3, 0.5]) == nn.hidden_layers[0].neurons[1].net)
    #     self.assertTrue(sigmoid_test(nn.hidden_layers[0].neurons[0].net) == nn.hidden_layers[0].neurons[0].out)
    #     self.assertTrue(sigmoid_test(nn.hidden_layers[0].neurons[1].net) == nn.hidden_layers[0].neurons[1].out)
    #     np.isclose(derivate_test(nn.output_layer.neurons[0]), nn.output_layer.neurons[0].fprime)
    #     np.isclose(derivate_test(nn.output_layer.neurons[1]), nn.output_layer.neurons[1].fprime)

    # def test_batch(self):
    #     self.single_train_p0()
    #     self.single_train_p1()
    #     self.train_batch()

    # def single_train_p0(self):
    #     nn = self.architecture(sigmoid)

    #     nn(1, 1)

    #     delta_out_0 = (0 - nn.output_layer.neurons[0].out) * nn.output_layer.neurons[0].fprime
    #     delta_out_1 = (1 - nn.output_layer.neurons[1].out) * nn.output_layer.neurons[1].fprime

    #     delta_hid_0 = (delta_out_0 * -1.5 + delta_out_1 * -0.5) * nn.hidden_layers[0].neurons[0].fprime
    #     delta_hid_1 = (delta_out_0 * 1.5 + delta_out_1 * 2) * nn.hidden_layers[0].neurons[1].fprime

    #     self.p0_delta_w_hid_00 = delta_hid_0
    #     self.p0_delta_w_hid_01 = delta_hid_0
    #     self.p0_delta_w_hid_02 = delta_hid_0
    #     self.p0_delta_w_hid_10 = delta_hid_1
    #     self.p0_delta_w_hid_11 = delta_hid_1
    #     self.p0_delta_w_hid_12 = delta_hid_1

    #     new_w_hid_00 = nn.hidden_layers[0].neurons[0].w[0] + 0.5 * self.p0_delta_w_hid_00
    #     new_w_hid_01 = nn.hidden_layers[0].neurons[0].w[1] + 0.5 * self.p0_delta_w_hid_01
    #     new_w_hid_02 = nn.hidden_layers[0].neurons[0].w[2] + 0.5 * self.p0_delta_w_hid_02
    #     new_w_hid_10 = nn.hidden_layers[0].neurons[1].w[0] + 0.5 * self.p0_delta_w_hid_10
    #     new_w_hid_11 = nn.hidden_layers[0].neurons[1].w[1] + 0.5 * self.p0_delta_w_hid_11
    #     new_w_hid_12 = nn.hidden_layers[0].neurons[1].w[2] + 0.5 * self.p0_delta_w_hid_12

    #     self.p0_delta_w_out_00 = delta_out_0 * 1
    #     self.p0_delta_w_out_01 = delta_out_0 * nn.hidden_layers[0].neurons[0].out
    #     self.p0_delta_w_out_02 = delta_out_0 * nn.hidden_layers[0].neurons[1].out
    #     self.p0_delta_w_out_10 = delta_out_1 * 1
    #     self.p0_delta_w_out_11 = delta_out_1 * nn.hidden_layers[0].neurons[0].out
    #     self.p0_delta_w_out_12 = delta_out_1 * nn.hidden_layers[0].neurons[1].out

    #     new_w_out_00 = nn.output_layer.neurons[0].w[0] + 0.5 * self.p0_delta_w_out_00
    #     new_w_out_01 = nn.output_layer.neurons[0].w[1] + 0.5 * self.p0_delta_w_out_01
    #     new_w_out_02 = nn.output_layer.neurons[0].w[2] + 0.5 * self.p0_delta_w_out_02
    #     new_w_out_10 = nn.output_layer.neurons[1].w[0] + 0.5 * self.p0_delta_w_out_10
    #     new_w_out_11 = nn.output_layer.neurons[1].w[1] + 0.5 * self.p0_delta_w_out_11
    #     new_w_out_12 = nn.output_layer.neurons[1].w[2] + 0.5 * self.p0_delta_w_out_12

    #     nn.train([([1, 1], [0, 1])])

    #     self.assertTrue(nn.hidden_layers[0].neurons[0].w[0] == new_w_hid_00)
    #     self.assertTrue(nn.hidden_layers[0].neurons[0].w[1] == new_w_hid_01)
    #     self.assertTrue(nn.hidden_layers[0].neurons[0].w[2] == new_w_hid_02)
    #     self.assertTrue(nn.hidden_layers[0].neurons[1].w[0] == new_w_hid_10)
    #     self.assertTrue(nn.hidden_layers[0].neurons[1].w[1] == new_w_hid_11)
    #     self.assertTrue(nn.hidden_layers[0].neurons[1].w[2] == new_w_hid_12)

    #     self.assertTrue(nn.output_layer.neurons[0].w[0] == new_w_out_00)
    #     self.assertTrue(nn.output_layer.neurons[0].w[1] == new_w_out_01)
    #     self.assertTrue(nn.output_layer.neurons[0].w[2] == new_w_out_02)
    #     self.assertTrue(nn.output_layer.neurons[1].w[0] == new_w_out_10)
    #     self.assertTrue(nn.output_layer.neurons[1].w[1] == new_w_out_11)
    #     self.assertTrue(nn.output_layer.neurons[1].w[2] == new_w_out_12)

    # def single_train_p1(self):
    #     nn = self.architecture(sigmoid)

    #     nn(2, 2)

    #     delta_out_0 = (1 - nn.output_layer.neurons[0].out) * nn.output_layer.neurons[0].fprime
    #     delta_out_1 = (1 - nn.output_layer.neurons[1].out) * nn.output_layer.neurons[1].fprime

    #     delta_hid_0 = (delta_out_0 * -1.5 + delta_out_1 * -0.5) * nn.hidden_layers[0].neurons[0].fprime
    #     delta_hid_1 = (delta_out_0 * 1.5 + delta_out_1 * 2) * nn.hidden_layers[0].neurons[1].fprime

    #     self.p1_delta_w_hid_00 = delta_hid_0 * 1
    #     self.p1_delta_w_hid_01 = delta_hid_0 * 2
    #     self.p1_delta_w_hid_02 = delta_hid_0 * 2
    #     self.p1_delta_w_hid_10 = delta_hid_1 * 1
    #     self.p1_delta_w_hid_11 = delta_hid_1 * 2
    #     self.p1_delta_w_hid_12 = delta_hid_1 * 2

    #     new_w_hid_00 = nn.hidden_layers[0].neurons[0].w[0] + 0.5 * self.p1_delta_w_hid_00
    #     new_w_hid_01 = nn.hidden_layers[0].neurons[0].w[1] + 0.5 * self.p1_delta_w_hid_01
    #     new_w_hid_02 = nn.hidden_layers[0].neurons[0].w[2] + 0.5 * self.p1_delta_w_hid_02
    #     new_w_hid_10 = nn.hidden_layers[0].neurons[1].w[0] + 0.5 * self.p1_delta_w_hid_10
    #     new_w_hid_11 = nn.hidden_layers[0].neurons[1].w[1] + 0.5 * self.p1_delta_w_hid_11
    #     new_w_hid_12 = nn.hidden_layers[0].neurons[1].w[2] + 0.5 * self.p1_delta_w_hid_12

    #     self.p1_delta_w_out_00 = delta_out_0 * 1
    #     self.p1_delta_w_out_01 = delta_out_0 * nn.hidden_layers[0].neurons[0].out
    #     self.p1_delta_w_out_02 = delta_out_0 * nn.hidden_layers[0].neurons[1].out
    #     self.p1_delta_w_out_10 = delta_out_1 * 1
    #     self.p1_delta_w_out_11 = delta_out_1 * nn.hidden_layers[0].neurons[0].out
    #     self.p1_delta_w_out_12 = delta_out_1 * nn.hidden_layers[0].neurons[1].out

    #     new_w_out_00 = nn.output_layer.neurons[0].w[0] + 0.5 * self.p1_delta_w_out_00
    #     new_w_out_01 = nn.output_layer.neurons[0].w[1] + 0.5 * self.p1_delta_w_out_01
    #     new_w_out_02 = nn.output_layer.neurons[0].w[2] + 0.5 * self.p1_delta_w_out_02
    #     new_w_out_10 = nn.output_layer.neurons[1].w[0] + 0.5 * self.p1_delta_w_out_10
    #     new_w_out_11 = nn.output_layer.neurons[1].w[1] + 0.5 * self.p1_delta_w_out_11
    #     new_w_out_12 = nn.output_layer.neurons[1].w[2] + 0.5 * self.p1_delta_w_out_12

    #     nn.train([([2, 2], [1, 1])])

    #     self.assertTrue(nn.hidden_layers[0].neurons[0].w[0] == new_w_hid_00)
    #     self.assertTrue(nn.hidden_layers[0].neurons[0].w[1] == new_w_hid_01)
    #     self.assertTrue(nn.hidden_layers[0].neurons[0].w[2] == new_w_hid_02)
    #     self.assertTrue(nn.hidden_layers[0].neurons[1].w[0] == new_w_hid_10)
    #     self.assertTrue(nn.hidden_layers[0].neurons[1].w[1] == new_w_hid_11)
    #     self.assertTrue(nn.hidden_layers[0].neurons[1].w[2] == new_w_hid_12)

    #     self.assertTrue(nn.output_layer.neurons[0].w[0] == new_w_out_00)
    #     self.assertTrue(nn.output_layer.neurons[0].w[1] == new_w_out_01)
    #     self.assertTrue(nn.output_layer.neurons[0].w[2] == new_w_out_02)
    #     self.assertTrue(nn.output_layer.neurons[1].w[0] == new_w_out_10)
    #     self.assertTrue(nn.output_layer.neurons[1].w[1] == new_w_out_11)
    #     self.assertTrue(nn.output_layer.neurons[1].w[2] == new_w_out_12)

    # def train_batch(self):
    #     self.assertTrue(self.p0_delta_w_out_00 != -1)
    #     self.assertTrue(self.p1_delta_w_out_00 != -1)
    #     self.assertTrue(self.p0_delta_w_out_01 != -1)
    #     self.assertTrue(self.p1_delta_w_out_01 != -1)
    #     self.assertTrue(self.p0_delta_w_out_02 != -1)
    #     self.assertTrue(self.p1_delta_w_out_02 != -1)
    #     self.assertTrue(self.p0_delta_w_out_10 != -1)
    #     self.assertTrue(self.p1_delta_w_out_10 != -1)
    #     self.assertTrue(self.p0_delta_w_out_11 != -1)
    #     self.assertTrue(self.p1_delta_w_out_11 != -1)
    #     self.assertTrue(self.p0_delta_w_out_12 != -1)
    #     self.assertTrue(self.p1_delta_w_out_12 != -1)
    #     self.assertTrue(self.p0_delta_w_hid_00 != -1)
    #     self.assertTrue(self.p1_delta_w_hid_00 != -1)
    #     self.assertTrue(self.p0_delta_w_hid_01 != -1)
    #     self.assertTrue(self.p1_delta_w_hid_01 != -1)
    #     self.assertTrue(self.p0_delta_w_hid_02 != -1)
    #     self.assertTrue(self.p1_delta_w_hid_02 != -1)
    #     self.assertTrue(self.p0_delta_w_hid_10 != -1)
    #     self.assertTrue(self.p1_delta_w_hid_10 != -1)
    #     self.assertTrue(self.p0_delta_w_hid_11 != -1)
    #     self.assertTrue(self.p1_delta_w_hid_11 != -1)
    #     self.assertTrue(self.p0_delta_w_hid_12 != -1)
    #     self.assertTrue(self.p1_delta_w_hid_12 != -1)

    #     nn = self.architecture(sigmoid)

    #     delta_w_out_00 = (self.p0_delta_w_out_00 + self.p1_delta_w_out_00) / 2
    #     delta_w_out_01 = (self.p0_delta_w_out_01 + self.p1_delta_w_out_01) / 2
    #     delta_w_out_02 = (self.p0_delta_w_out_02 + self.p1_delta_w_out_02) / 2
    #     delta_w_out_10 = (self.p0_delta_w_out_10 + self.p1_delta_w_out_10) / 2
    #     delta_w_out_11 = (self.p0_delta_w_out_11 + self.p1_delta_w_out_11) / 2
    #     delta_w_out_12 = (self.p0_delta_w_out_12 + self.p1_delta_w_out_12) / 2
    #     delta_w_hid_00 = (self.p0_delta_w_hid_00 + self.p1_delta_w_hid_00) / 2
    #     delta_w_hid_01 = (self.p0_delta_w_hid_01 + self.p1_delta_w_hid_01) / 2
    #     delta_w_hid_02 = (self.p0_delta_w_hid_02 + self.p1_delta_w_hid_02) / 2
    #     delta_w_hid_10 = (self.p0_delta_w_hid_10 + self.p1_delta_w_hid_10) / 2
    #     delta_w_hid_11 = (self.p0_delta_w_hid_11 + self.p1_delta_w_hid_11) / 2
    #     delta_w_hid_12 = (self.p0_delta_w_hid_12 + self.p1_delta_w_hid_12) / 2

    #     new_w_out_00 = nn.output_layer.neurons[0].w[0] + 0.5 * delta_w_out_00
    #     new_w_out_01 = nn.output_layer.neurons[0].w[1] + 0.5 * delta_w_out_01
    #     new_w_out_02 = nn.output_layer.neurons[0].w[2] + 0.5 * delta_w_out_02
    #     new_w_out_10 = nn.output_layer.neurons[1].w[0] + 0.5 * delta_w_out_10
    #     new_w_out_11 = nn.output_layer.neurons[1].w[1] + 0.5 * delta_w_out_11
    #     new_w_out_12 = nn.output_layer.neurons[1].w[2] + 0.5 * delta_w_out_12
    #     new_w_hid_00 = nn.hidden_layers[0].neurons[0].w[0] + 0.5 * delta_w_hid_00
    #     new_w_hid_01 = nn.hidden_layers[0].neurons[0].w[1] + 0.5 * delta_w_hid_01
    #     new_w_hid_02 = nn.hidden_layers[0].neurons[0].w[2] + 0.5 * delta_w_hid_02
    #     new_w_hid_10 = nn.hidden_layers[0].neurons[1].w[0] + 0.5 * delta_w_hid_10
    #     new_w_hid_11 = nn.hidden_layers[0].neurons[1].w[1] + 0.5 * delta_w_hid_11
    #     new_w_hid_12 = nn.hidden_layers[0].neurons[1].w[2] + 0.5 * delta_w_hid_12

    #     nn.train([([1, 1], [0, 1]), ([2, 2], [1, 1])])

    #     self.assertTrue(new_w_out_00 == nn.output_layer.neurons[0].w[0])
    #     self.assertTrue(new_w_out_01 == nn.output_layer.neurons[0].w[1])
    #     self.assertTrue(new_w_out_02 == nn.output_layer.neurons[0].w[2])
    #     self.assertTrue(new_w_out_10 == nn.output_layer.neurons[1].w[0])
    #     self.assertTrue(new_w_out_11 == nn.output_layer.neurons[1].w[1])
    #     self.assertTrue(new_w_out_12 == nn.output_layer.neurons[1].w[2])
    #     self.assertTrue(new_w_hid_00 == nn.hidden_layers[0].neurons[0].w[0])
    #     self.assertTrue(new_w_hid_01 == nn.hidden_layers[0].neurons[0].w[1])
    #     self.assertTrue(new_w_hid_02 == nn.hidden_layers[0].neurons[0].w[2])
    #     self.assertTrue(new_w_hid_10 == nn.hidden_layers[0].neurons[1].w[0])
    #     self.assertTrue(new_w_hid_11 == nn.hidden_layers[0].neurons[1].w[1])
    #     self.assertTrue(new_w_hid_12 == nn.hidden_layers[0].neurons[1].w[2])

    def test_batch_explicit(self):
        nn = MultilayerPerceptron(
                size_input_layer=2,
                size_output_layer=2,
                sizes_hidden_layers=[2],
                hidden_weights=[[[0, 1.5, 2], [0, 3, 0.5]]],
                output_weights=[[0, -1.5, 1.5], [0, -0.5, 2]]
        )(sigmoid)

        nn.train([([1, 1], [0, 1]), ([2, 2], [1, 1])])

        self.assertTrue(np.isclose(
            nn.output_layer.w,
            [[0., -1.49911246, 1.50088754], [0.01406306, -0.48615559, 2.01384441]]
        ).all())

        self.assertTrue(np.isclose(
            nn.hidden_layers[0].w,
            [[1.18486022e-03, 1.50113909e+00, 2.00113909e+00], [-8.66234240e-04, 2.99918884e+00, 4.99188840e-01]]
        ).all())


if __name__ == '__main__':
    unittest.main()
