from __future__ import annotations
import sympy as sp
from typing import Union


class ActivationFunction:
    def __init__(self, exepression: Union[ActivationFunction, sp.Expr, str]):
        self._expr: sp.Expr
        if isinstance(exepression, ActivationFunction):
            self._expr = exepression._expr
        elif isinstance(exepression, sp.Expr):
            self._expr = exepression
        elif isinstance(exepression, str):
            self._expr = sp.sympify(exepression)
        else:
            raise TypeError('expression')

        symbols = self._expr.atoms(sp.Symbol)
        if len(symbols) != 1:
            raise ValueError('expression must have exactly one parameter')

        x = list(symbols)[0]
        self._function = sp.lambdify(x, self._expr, 'numpy')

        derivative = self._expr.diff(x)
        if isinstance(derivative, sp.Derivative):
            self._derivative = None
        else:
            self._derivative = sp.lambdify(x, derivative, 'numpy')

    def __call__(self, x: float) -> float:
        return self._function(x)

    def isdifferentiable(self) -> bool:
        return True if self._derivative else False

    def derivative(self, x: float) -> float:
        if not self.isdifferentiable():
            raise RuntimeError('this function is not differentiable')

        return self._derivative(x)

    def __repr__(self) -> str:
        return self._expr.__repr__()

    def __str__(self) -> str:
        return self._expr.__str__()

    def __eq__(self,  other):
        if not isinstance(other, ActivationFunction):
            return False
        return self._expr == other._expr

    def __hash__(self):
        return hash((self._expr))


identity = ActivationFunction('x')

sign = ActivationFunction('x>0')


def sigmoidal_a(a: float) -> ActivationFunction:
    return ActivationFunction(sp.sympify('1/(1+exp(-a*x))').subs('a', a))


sigmoidal_1 = sigmoidal_a(1)
