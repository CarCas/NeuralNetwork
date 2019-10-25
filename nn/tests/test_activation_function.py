import unittest
import numpy as np
import sympy as sp

from nn.activation_function import (
    ActivationFunction,
    identity,
    sign,
    sigmoidal_a)


class TestActivationFunction(unittest.TestCase):
    def test_type(self):
        with self.assertRaises(TypeError):
            ActivationFunction(lambda x: x)

    def test_value(self):
        with self.assertRaises(ValueError):
            ActivationFunction('x+y')

    def test_identity(self):
        self.assertEqual(identity(0), 0)
        self.assertEqual(identity(1), 1)
        self.assertEqual(identity(7), 7)

        self.assertTrue(identity.isdifferentiable())
        self.assertEqual(identity.derivative(0), 1)
        self.assertEqual(identity.derivative(1), 1)
        self.assertEqual(identity.derivative(7), 1)

    def test_sign(self):
        self.assertEqual(sign(0), 0)
        self.assertEqual(sign(1), 1)
        self.assertEqual(sign(-1), 0)

        self.assertFalse(sign.isdifferentiable())
        with self.assertRaises(RuntimeError):
            sign.derivative(1)

    def test_sigmoidal_a(self):
        sigmoidal_2 = sigmoidal_a(2)

        self.assertEqual(sigmoidal_2(0), 1/2)
        self.assertAlmostEqual(sigmoidal_2(1), np.exp(2)/(1+np.exp(2)))
        self.assertAlmostEqual(sigmoidal_2(2), np.exp(4)/(1+np.exp(4)))

        self.assertTrue(sigmoidal_2.isdifferentiable())
        self.assertEqual(sigmoidal_2.derivative(0), 1/2)
        # from wolframalpha
        self.assertAlmostEqual(sigmoidal_2.derivative(1), 0.209987170807013)
        self.assertEqual(sigmoidal_2.derivative(2), 0.03532541242658223)

    def test_init(self):
        fs = [
            ActivationFunction('x**4'),
            ActivationFunction(sp.simplify('x**4')),
            ActivationFunction(ActivationFunction('x**4'))
        ]

        for f in fs:
            self.assertEqual(f(0), 0)
            self.assertEqual(f(1), 1)
            self.assertEqual(f(2), 16)
            self.assertEqual(f(3), 81)

            self.assertTrue(f.isdifferentiable())
            d = f.derivative
            self.assertEqual(d(0), 0)
            self.assertEqual(d(1), 4*(1**3))
            self.assertEqual(d(2), 4*(2**3))
            self.assertEqual(d(3), 4*(3**3))

    def test_custom_expr(self):
        f = ActivationFunction(sp.simplify('x**4'))
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 16)
        self.assertEqual(f(3), 81)

        self.assertTrue(f.isdifferentiable())
        d = f.derivative
        self.assertEqual(d(0), 0)
        self.assertEqual(d(1), 4*(1**3))
        self.assertEqual(d(2), 4*(2**3))
        self.assertEqual(d(3), 4*(3**3))


if __name__ == '__main__':
    unittest.main()
