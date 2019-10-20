import numpy as np


# LOSS FUNCTIONS
class LossFunction:
    def __init__(self, loss):
        self.loss = loss

    @staticmethod
    def mean_square_loss_fun(y, x):
        return np.mean(np.square(np.subtract(y, x)))

    @staticmethod
    def mean_square_loss_der(y, x):
        return np.multiply(2, np.subtract(y, x))

    @staticmethod
    def root_mean_squared_loss_fun(y, x):
        return np.sqrt(np.mean(np.square(np.subtract(y, x))))

    @staticmethod
    def root_mean_squared_loss_der(y, x):
        num = np.multiply(2, np.subtract(y, x))
        den = np.multiply(2, np.sqrt(np.mean(np.square(np.subtract(y, x)))))
        np.divide(num, den)

    @staticmethod
    def mean_euclidean_loss_fun(y, x):
        sbtr = np.subtract(y, x)
        return np.sqrt(np.dot(sbtr, sbtr))

    @staticmethod
    def mean_euclidean_loss_der(y, x):
        sbtr = np.subtract(y, x)
        return np.divide(sbtr, np.sqrt(np.dot(sbtr, sbtr)))

    @staticmethod
    def mean_absolute_loss_fun(y, x):
        return np.mean(np.absolute(np.subtract(y, x)))

    @staticmethod
    def mean_absolute_loss_der(y, x):
        sbtr = np.subtract(y, x)
        return np.divide(sbtr, np.absolute(sbtr))

    def apply(self, opt, y, x):
        """
        :type opt: basestring
        :param opt: option to choose
        :type y: array
        :type x: array
        :rtype: (number, number)
        :return:
        """
        options = {
            "MSE": (self.mean_square_loss_fun(y, x), self.mean_square_loss_der(y, x)),
            "MEE": (self.mean_euclidean_loss_fun(y, x), self.mean_euclidean_loss_der(y, x)),
            "MAE": (self.mean_absolute_loss_fun(y, x), self.mean_absolute_loss_der(y, x)),
            "RMSE": (self.root_mean_squared_loss_fun(y, x), self.root_mean_squared_loss_der(y, x))
        }
        return options[opt]


# ACTIVATION FUNCTIONS
class ActivationFunction:
    def __init__(self, act):
        self.act = act

    def logistic_fun(self, x):
        return np.divide(1., np.add(1., np.exp(np.multiply(-1, x))))

    def logistic_der(self, x):
        return np.multiply(self.logistic_fun(x), (np.subtract(1, self.logistic_fun(x))))

    def relu_fun(self, x):
        x[x <= 0] = 0
        return x

    def relu_der(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    # parametric relu
    def prelu_fun(self, x, alpha):
        x[x <= 0] = np.multiply(alpha, x)
        return x

    def prelu_der(self, x, alpha):
        x[x <= 0] = alpha
        x[x > 0] = 1
        return x

    def soft_plus_fun(self, x):
        return np.log(np.add(1, np.exp(x)))

    def soft_plus_der(self, x):
        return np.power(np.add(1, np.exp(np.multiply(-1, x))), -1)

    def gaussian_fun(self, x):
        return np.exp(np.multiply(-1, np.square(x)))

    def gaussian_der(self, x):
        return np.multiply(np.multiply(-2, x), np.exp(np.multiply(-1, np.square(x, 2))))

    def apply(self, opt, z, alpha):
        options = {
            "logistic": (self.logistic_fun(z), self.logistic_der(z)),
            "relu": (self.relu_fun(z), self.relu_der(z)),
            "prelu": (self.prelu_fun(z, alpha), self.prelu_der(z, alpha)),
            "soft_plus": (self.soft_plus_fun(z), self.soft_plus_der(z)),
            "gaussian": (self.gaussian_fun(z), self.gaussian_der(z))
        }
        return options[opt]


class Layer:

    def __init__(self, in_degree, out_degree, activation_function, mu):
        self.w = np.random.randn(out_degree, in_degree)
        self.b = np.random.randn(out_degree)
        self.activation_function = activation_function
        self.mu = mu

    def feed_forward(self, x):
        self.x = x
        self.z = np.dot(self.w, self.x) + self.b

        return self.activation_function(self.z)[0]

    def back_prop(self, node):
        return np.dot(np.dot(node, self.activation_function(self.z)), self.w)

class NeuralNetwork:
    weights = []
    layers = []

    def __init__(self, x, y, hls_dim: [int]):
        self.input = x
        self.y = y
        self.weights.append(np.random.rand(self.input.shape[1], hls_dim[0]))
        for i in range(len(hls_dim) - 1):
            self.weights.append(np.random.rand(hls_dim[i + 1], hls_dim[i]))
        self.output = np.zeros(y.shape)

    def feedforward(self, activation_function):
        self.layers.append(activation_function(np.dot(self.input, self.weights[0])))
        l = len(self.weights)
        for i in range(l - 1):
            self.layers.append(activation_function(np.dot(self.weights[i], self.weights[i + 1])))
        self.output = activation_function(np.dot(self.weights[l - 2], self.weights[l - 1]))

    # not implemented yet - we need to define the chain rule for the MEE - found chain rule for SSE
    def backprop(self):
        x = ()
