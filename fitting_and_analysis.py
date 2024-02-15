import numpy as np
import scipy.stats as spstats
from decimal import Decimal

class CurveFitFuncs():
    def __init__(self):
        pass

    def remove_systematic_error(self, arr):
        return arr - arr[0]

    def residual(self, yarr_measured, yarr_predicted):
        return yarr_measured - yarr_predicted

    def sum_squared_ratio(self, numer, denom):
        return np.sum((numer ** 2) / (denom ** 2))

    def calc_dof(self, yarr_measured, params_in_model):
        dof = len(yarr_measured) - params_in_model
        return dof

    def calc_raw_chi_squared(self, yarr_measured, yarr_predicted, y_uncertainty):
        numer = self.residual(yarr_measured, yarr_predicted)
        denom = y_uncertainty
        return self.sum_squared_ratio(numer, denom)

    def calc_reduced_chi_squared(self, yarr_measured, yarr_predicted, y_uncertainty, params_in_model):
        numer = self.residual(yarr_measured, yarr_predicted)
        denom = y_uncertainty
        dof = len(yarr_measured) - params_in_model
        return self.sum_squared_ratio(numer, denom) / dof

    def calc_chi2_probability(self, raw_chi2, dof):
        chi2_prob = (1 - spstats.chi2.cdf(raw_chi2, dof))
        return chi2_prob


class CurveFitAnalysis():
    def __init__(self, xarr, yarr_measured, yarr_uncertainty, FittedFunc): # FittedFunc must have number_of_parameters attribute
        cff = CurveFitFuncs()
        yarr_predicted = FittedFunc(xarr)

        self.degrees_of_freedom = cff.calc_dof(yarr_measured, FittedFunc.number_of_parameters)
        self.raw_chi2 = cff.calc_raw_chi_squared(yarr_measured, yarr_predicted, yarr_uncertainty)
        self.reduced_chi2 = cff.calc_reduced_chi_squared(yarr_measured, yarr_predicted, yarr_uncertainty, FittedFunc.number_of_parameters)
        self.chi2_probability = cff.calc_chi2_probability(self.raw_chi2, self.degrees_of_freedom)



class Output():
    def __init__(self):
        pass

    def baseplot_errorbars(self, ax, x, y, yerr=None, xerr=None, **kwargs):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle='None', capsize=2, **kwargs)

    def baseplot_errorbars_with_markers(self, ax, x, y, yerr=None, xerr=None, **kwargs):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle='None', capsize=2, marker='.', **kwargs)

    def get_dp(self, num): # returns number of decimal places
        decimal = Decimal(str(num))
        try:
            if decimal.as_tuple().exponent >= 0:
                dp = -int(np.log10(float(num)))
            else:
                dp = -decimal.as_tuple().exponent
        except:
            dp = 0
        return int(dp)

    def to_sf(self, num, sf=3, **kwargs):
        result = np.format_float_positional(num, precision=sf, fractional=False, min_digits=sf, **kwargs)
        return result

    def get_leading_dp(self, num): # returns dp of first sig fig
        num = self.to_sf(num=num, sf=1)
        dp = self.get_dp(num)
        return int(dp)

    def print_scientific(self, num):
        dp = self.get_leading_dp(num=num)

        num_significant = Decimal(num) * 10 ** Decimal(dp)
        num_significant = round(num_significant, dp)
        num_significant = np.format_float_positional(float(num_significant), trim='-')

        if dp == 0:
            string = r'$%s$' %(num_significant)
        else:
            string = r'$%s \times 10^{%s}$' %(num_significant, -dp)

        return string


    def print_with_uncertainty(self, num, uncertainty):
        rounded_uncertainty = self.to_sf(uncertainty, sf=1)
        uncertainty_dp = self.get_dp(rounded_uncertainty)

        rounded_num = round(num, uncertainty_dp)

        rounded_num_leading_dp = self.get_leading_dp(rounded_num)

        num_significant = Decimal(rounded_num) * 10 ** Decimal(rounded_num_leading_dp)
        uncertainty_significant = Decimal(rounded_uncertainty) * 10 ** Decimal(rounded_num_leading_dp)

        num_significant = round(num_significant, self.get_dp(uncertainty_significant))

        num_significant = np.format_float_positional(float(num_significant), trim='-')
        uncertainty_significant = np.format_float_positional(float(uncertainty_significant), trim='-')

        if '.' in uncertainty_significant and '.' not in num_significant and len(num_significant) < len(uncertainty_significant):
            num_significant += '.'

        while len(num_significant) < len(uncertainty_significant):
            num_significant += '0'

        if rounded_num_leading_dp == 0:
            string = r'$%s \pm %s$' %(num_significant, uncertainty_significant)
        else:
            string = r'$(%s \pm %s) \times 10^{%s}$' %(num_significant, uncertainty_significant, -rounded_num_leading_dp)
        return string