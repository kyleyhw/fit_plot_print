import numpy as np
import scipy.odr
from scipy.optimize import curve_fit
from matplotlib import rc
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
rc('font', **font)
from matplotlib.offsetbox import AnchoredText

from fitting_and_analysis import CurveFitFuncs
from fitting_and_analysis import CurveFitAnalysis
from fitting_and_analysis import Output
Output = Output()

class Fitting():
    def __init__(self, model, x, y_measured, y_error, x_error=None, p0=None, bounds=(-np.inf, np.inf), units_for_parameters=None, method='lr'): # requires hard coding for now
        self.model = model
        self.x = x
        self.x_error = x_error
        self.y_measured = y_measured
        self.y_error = y_error

        if method == 'lr':
            self.popt, self.pcov = curve_fit(self.model, self.x, self.y_measured, sigma=y_error, absolute_sigma=True, p0=p0, bounds=bounds, maxfev=1000000)
            self.parameter_errors = np.sqrt(np.diag(self.pcov))

        elif method == 'odr':
            odr_data = scipy.odr.RealData(x=self.x, y=self.y_measured, sx=self.x_error, sy=self.y_error)
            odr_model = scipy.odr.Model(lambda beta, x : self.model(x, *beta))

            fitted = scipy.odr.ODR(odr_data, odr_model, beta0=p0, maxit=1000000)

            odr_output = fitted.run()
            self.popt = odr_output.beta
            self.pcov = odr_output.cov_beta
            self.parameter_errors = odr_output.sd_beta

        else:
            raise Exception('invalid fitting method')

        self.fitted_function = self.model.CorrespondingFittedFunction(popt=self.popt, parameter_errors=self.parameter_errors, units_for_parameters=units_for_parameters)

        self.optimal_parameters = {self.fitted_function.parameter_names[i]: self.popt[i] for i in range(len(self.fitted_function.parameter_names))}
        self.error_in_parameters = {self.fitted_function.parameter_names[i]: self.popt[i] for i in range(len(self.fitted_function.parameter_names))} # terrible variable naming

        self.y_predicted = self.fitted_function(self.x)

        self.cfa = CurveFitAnalysis(self.x, self.y_measured, self.y_error, self.fitted_function)

    def generate_anchor_text(self, info_sigfigs=3):
        raw_chi2_text = Output.to_sf(self.cfa.raw_chi2, sf=info_sigfigs, trim='-')
        if self.cfa.chi2_probability == 0:
            chi2_prob_text = ' < ' + Output.print_scientific(
                float(Output.to_sf(np.finfo(self.cfa.chi2_probability).eps, sf=info_sigfigs, trim='-')))
        else:
            if Output.get_dp(Output.to_sf(self.cfa.chi2_probability, sf=info_sigfigs, trim='k')) > 5:
                chi2_prob_text = ' = ' + Output.print_scientific(
                    float(Output.to_sf(self.cfa.chi2_probability, sf=info_sigfigs, trim='k')))
            else:
                chi2_prob_text = ' = ' + str(
                    float(Output.to_sf(self.cfa.chi2_probability, sf=info_sigfigs, trim='k')))

        text = self.fitted_function.parameter_info + \
                         '\n$\chi^2$ / DOF = ' + raw_chi2_text + ' / ' + str(self.cfa.degrees_of_freedom) + ' = ' + str(Output.to_sf(self.cfa.reduced_chi2, sf=info_sigfigs)) + \
                         '\n$\chi^2$ prob' + chi2_prob_text

        return text

    def scatter_plot_data_and_fit(self, ax, plot_fit=True, info_loc='upper left', legend_loc='upper right', **kwargs):

        Output.baseplot_errorbars_with_markers(ax=ax, x=self.x, y=self.y_measured, yerr=self.y_error, xerr=self.x_error, label='data', **kwargs)

        if plot_fit:
            x_for_plotting_fit = np.linspace(min(self.x), max(self.x), 10000)
            y_for_plotting_fit = np.zeros_like(x_for_plotting_fit) + self.fitted_function(x_for_plotting_fit)

            ax.plot(x_for_plotting_fit, y_for_plotting_fit, label='fit', linewidth=2, alpha=1)

            info_sigfigs = 3
            info_fontsize = 14

            info_on_ax = self.generate_anchor_text(info_sigfigs=info_sigfigs)

            ax_text = AnchoredText(info_on_ax, loc=info_loc, frameon=False, prop=dict(fontsize=info_fontsize))
            ax.add_artist(ax_text)

        ax.legend(loc=legend_loc)
        ax.grid()

        self.data_plot_xlim = ax.get_xlim()

    def plot_residuals(self, ax, **kwargs):
        cff = CurveFitFuncs()

        residuals = cff.residual(self.y_measured, self.y_predicted)
        error_in_residuals = self.y_error # np.sqrt((self.parameter_errors[0] * self.x)**2 + (self.y_error)**2)


        Output.baseplot_errorbars_with_markers(ax=ax, x=self.x, y=residuals, yerr=error_in_residuals, xerr=None,
                                               label='residuals', **kwargs)

        ax.set_xlim(*self.data_plot_xlim)
        ax.axhline(linewidth=1, color='k')
        ax.grid()

        ax.legend(loc='upper right')