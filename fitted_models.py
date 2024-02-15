import numpy as np

import fit_models
from fitting_and_analysis import Output

Output = Output()


class GaussianWithBase:  # height is usually 1; not probability density
    def __init__(self, popt, parameter_errors, units_for_parameters, info_sigfigs=2):
        self.popt = popt
        (self.base, self.scale, self.mu, self.sigma) = popt

        self.function = fit_models.GaussianWithBase()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('base', 'scale', 'mu', 'sigma')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result


class GaussianZeroCenter:  # height is usually 1; not probability density
    def __init__(self, popt, parameter_errors, info_sigfigs=2):
        self.popt = popt
        (self.base, self.scale, self.sigma) = popt

        self.function = fit_models.GaussianZeroCenter()
        self.number_of_parameters = self.function.number_of_parameters

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result


class Linear:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.m, self.c) = popt
        (self.m_error, self.c_error) = parameter_errors

        self.function = fit_models.Linear()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('m', 'c')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class Proportional:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.m) = popt[0]
        (self.m_error) = parameter_errors[0]

        self.function = fit_models.Proportional()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('m')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = ' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ' ' +
                                    self.units_for_parameters for i in range(self.number_of_parameters)] # removed index for parameter units so that it uses the whole string; otherwise, only takes first char of str
        self.parameter_info = '\n'.join(self.parameter_info_list)


    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class Constant:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.c) = popt[0]
        (self.c_error) = parameter_errors[0]

        self.function = fit_models.Constant()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('c')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class QuadraticMonomial:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.c) = popt[0]
        (self.c_error) = parameter_errors[0]

        self.function = fit_models.QuadraticMonomial()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('a')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class SquareRootProportional:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.c) = popt[0]
        (self.c_error) = parameter_errors[0]

        self.function = fit_models.SquareRootProportional()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('a')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class QuadraticPlusProportionalMonomials:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.a, self.b) = popt
        (self.a_error, self.b_error) = parameter_errors

        self.function = fit_models.QuadraticPlusProportionalMonomials()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('a', 'b')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class DecayingSinusoid:
    def __init__(self, popt, parameter_errors, units_for_parameters=None):
        self.popt = popt
        (self.B, self.A, self.T, self.tau, self.phi) = popt
        (self.B_error, self.A_error, self.T_error, self.tau_error, self.phi_error) = parameter_errors

        self.function = fit_models.DecayingSinusoid()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('B', 'A', 'T', r'$\phi$', r'$\frac{1}{\tau}$')
        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, t):
        result = self.function(t, *self.popt)
        return result

class GaussianNoBase:  # height is usually 1; not probability density
    def __init__(self, popt, parameter_errors, units_for_parameters, info_sigfigs=2):
        self.popt = popt
        (self.scale, self.mu, self.sigma) = popt

        self.function = fit_models.GaussianNoBase()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('scale', 'mu', 'sigma')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class Lognormal:  # height is usually 1; not probability density
    def __init__(self, popt, parameter_errors, units_for_parameters, info_sigfigs=2):
        self.popt = popt
        (self.scale, self.mu, self.sigma) = popt

        self.function = fit_models.Lognormal()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('scale', 'mu', 'sigma')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class Laplacian:  # height is usually 1; not probability density
    def __init__(self, popt, parameter_errors, units_for_parameters, info_sigfigs=2):
        self.popt = popt
        (self.scale, self.mu, self.sigma) = popt

        self.function = fit_models.Laplacian()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('scale', 'mu', 'sigma')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class Exponential:
    def __init__(self, popt, parameter_errors, units_for_parameters, info_sigfigs=2):
        self.popt = popt
        (self.scale, self.exponent) = popt

        self.function = fit_models.Exponential()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('scale', 'exponent')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class ComptonSigma:
    def __init__(self, popt, parameter_errors, units_for_parameters, info_sigfigs=2):
        self.popt = popt
        (self.b, self.c) = popt

        self.function = fit_models.ComptonSigma()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('b', 'c')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

# class ExponentialFixedIntercept:
#     def __init__(self, popt, parameter_errors, units_for_parameters, info_sigfigs=2):
#         self.popt = popt
#         (self.exponent) = popt[0]
#
#         self.function = fit_models.ExponentialFixedIntercept()
#         self.number_of_parameters = self.function.number_of_parameters
#         self.parameter_names = ['exponent']
#         self.units_for_parameters = units_for_parameters
#
#         if units_for_parameters == None:
#             self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
#         else:
#             self.units_for_parameters = units_for_parameters
#
#         self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
#                                                                                                           parameter_errors[
#                                                                                                               i]) + ') ' +
#                                     self.units_for_parameters[i] for i in range(self.number_of_parameters)]
#         self.parameter_info = '\n'.join(self.parameter_info_list)
#
#     def __call__(self, x):
#         result = self.function(x, *self.popt)
#         return result