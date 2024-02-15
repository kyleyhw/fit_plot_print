import numpy as np
import fitted_models


class GaussianWithBase:  # height is usually 1; not probability density
    def __init__(self):
        self.number_of_parameters = 4
        self.CorrespondingFittedFunction = fitted_models.GaussianWithBase

    def __call__(self, x, base, scale, mu, sigma):
        result = base + scale * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return result


class GaussianZeroCenter:  # height is usually 1; not probability density
    def __init__(self):
        self.number_of_parameters = 3
        self.CorrespondingFittedFunction = fitted_models.GaussianZeroCenter

    def __call__(self, x, base, scale, sigma):
        result = base + scale * np.exp(-0.5 * ((x) / sigma) ** 2)
        return result


class Linear:
    def __init__(self):
        self.number_of_parameters = 2
        self.CorrespondingFittedFunction = fitted_models.Linear

    def __call__(self, x, m, c):
        result = m * x + c
        return result

class Proportional:
    def __init__(self):
        self.number_of_parameters = 1
        self.CorrespondingFittedFunction = fitted_models.Proportional

    def __call__(self, x, m):
        result = m * x
        return result

class Constant:
    def __init__(self):
        self.number_of_parameters = 1
        self.CorrespondingFittedFunction = fitted_models.Constant

    def __call__(self, x, c):
        result = c
        return result

class QuadraticMonomial:
    def __init__(self):
        self.number_of_parameters = 1
        self.CorrespondingFittedFunction = fitted_models.QuadraticMonomial

    def __call__(self, x, a):
        result = a * x**2
        return result

class SquareRootProportional:
    def __init__(self):
        self.number_of_parameters = 1
        self.CorrespondingFittedFunction = fitted_models.SquareRootProportional

    def __call__(self, x, a):
        result = a * x**(1/2)
        return result

class QuadraticPlusProportionalMonomials:
    def __init__(self):
        self.number_of_parameters = 2
        self.CorrespondingFittedFunction = fitted_models.QuadraticPlusProportionalMonomials

    def __call__(self, x, a, b):
        result = (a * x**2) + (b * x)
        return result

class DecayingSinusoid:
    def __init__(self):
        self.number_of_parameters = 5
        self.CorrespondingFittedFunction = fitted_models.DecayingSinusoid
        # self.parameter_bounds = ([-np.inf, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, 2*np.pi, np.inf])

    def __call__(self, t, base, amplitude, period, phase, exponential_factor):
        result = base + np.exp(-t * exponential_factor) * amplitude * np.sin((2 * np.pi / period) * t + phase)
        return result

class GaussianNoBase:  # height is usually 1; not probability density
    def __init__(self):
        self.number_of_parameters = 3
        self.CorrespondingFittedFunction = fitted_models.GaussianNoBase

    def __call__(self, x, scale, mu, sigma):
        result = scale * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return result

class Lognormal:  # height is usually 1; not probability density
    def __init__(self):
        self.number_of_parameters = 3
        self.CorrespondingFittedFunction = fitted_models.Lognormal

    def __call__(self, x, scale, mu, sigma):
        result = (scale / (x * sigma)) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
        return result

class Laplacian:  # height is usually 1; not probability density
    def __init__(self):
        self.number_of_parameters = 3
        self.CorrespondingFittedFunction = fitted_models.Laplacian

    def __call__(self, x, scale, mu, sigma):
        result = scale * np.exp(-(np.abs(x - mu)) / sigma)
        return result

class Exponential:
    def __init__(self):
        self.number_of_parameters = 2
        self.CorrespondingFittedFunction = fitted_models.Exponential

    def __call__(self, x, scale, exponent):
        result = scale * np.exp(exponent * x)
        return result

class ComptonSigma:
    def __init__(self):
        self.number_of_parameters = 2
        self.CorrespondingFittedFunction = fitted_models.ComptonSigma

    def __call__(self, Z, b, c):
        result = b * Z**(4.2) + c * Z
        return result

# class ExponentialFixedIntercept:
#     def __init__(self):
#         self.number_of_parameters = 1
#         self.CorrespondingFittedFunction = fitted_models.ExponentialFixedIntercept
#
#     def __call__(self, x, exponent):
#
#         result =  * np.exp(exponent * x)
#         return result