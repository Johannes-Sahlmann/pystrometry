"""Utility functions to support DU437 activities.

Authors
-------

    Johannes Sahlmann

"""

import copy
import logging
import os

from astropy.table import Table
from astropy.time import Time
import numpy as np
import pandas as pd
import pylab as pl

from .. import gaia_astrometry, pystrometry
from ..pystrometry import geometric_elements

from uncertainties import unumpy as unp
from uncertainties import correlated_values_norm

try:
    from helpers.table_helpers import plot_columns_simple
except ImportError:
    print('universal_helpers not available')


@pd.api.extensions.register_dataframe_accessor("nss")
class NssDataFrame:
    """Extend the pandas DataFrame class for NSS tables.

    References
    ----------
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-extension-types

    """


    def __init__(self, pandas_obj):
        """Set _obj attribute and validate input dataframe."""
        self._obj = pandas_obj

    def add_geometric_elements(self, post_process=True):
        # self._obj = pd.concat([self._obj, self._obj.apply(lambda x: pystrometry.geometric_elements(np.array(
        #     [x['a_thiele_innes'], x['b_thiele_innes'], x['f_thiele_innes'], x['g_thiele_innes']])),
        #          axis=1, result_type='expand').rename(
        #     columns={0: 'p1_a1_mas', 1: 'p1_omega_deg', 2: 'p1_OMEGA_deg', 3: 'p1_incl_deg'})], axis='columns')

        thiele_innes_parameters = ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']
        thiele_innes_array = self._obj[thiele_innes_parameters].to_numpy().T
        geometric_parameters = geometric_elements(thiele_innes_array, post_process=post_process)
        # logging.debug(self._obj['source_id'].info())
        # concat converts source_id field from int64 to float64 (because some GE rows are NaN). avoid this
        self._obj[['p1_a1_mas', 'p1_omega_deg', 'p1_OMEGA_deg', 'p1_incl_deg']] = geometric_parameters.T
        # ge_df = pd.DataFrame(geometric_parameters.T, columns=['p1_a1_mas', 'p1_omega_deg', 'p1_OMEGA_deg', 'p1_incl_deg'])
        # self._obj = pd.concat([self._obj, ge_df], axis='columns')
        # logging.debug(self._obj['source_id'].info())
        if 'period' in self._obj.columns:
            self._obj.loc[:, 'p1_period_day'] = self._obj.loc[:, 'period']
        return self._obj

    def add_geometric_elements_unc(self):
        """ Add GE with uncertainties assuming Gaussian distributions and linear propagation."""

        def compute_ge(corr_vec, a,b,f,g,sa,sb,sf,sg):
            # Extract the submatrix of the correlation matrix corresponding to the thiele innes parameters
            correlation_matrix_ti = correlation_matrix(corr_vec)[5:9, 5:9]
            thiele_innes_parameters = np.array([a,b,f,g])
            thiele_innes_parameters_errors = np.array([sa,sb,sf,sg])
            ge1, ge1_err = geometric_elements_with_uncertainties(thiele_innes_parameters,
                                                                 thiele_innes_parameters_errors,
                                                                 correlation_matrix_ti,
                                                                 post_process=True,
                                                                 return_angles_in_deg=True)
            return np.concatenate((ge1, ge1_err))

        if ('p1_a1_mas' not in self._obj.columns) and ('sigma_p1_a1_mas' not in self._obj.columns):
        # if ('sigma_p1_a1_mas' not in self._obj.columns):
            tmp = self._obj.apply(lambda x: compute_ge(x['corr_vec'], x['a_thiele_innes'], x['b_thiele_innes'], x['f_thiele_innes'], x['g_thiele_innes'],
                                                       x['a_thiele_innes_error'], x['b_thiele_innes_error'], x['f_thiele_innes_error'], x['g_thiele_innes_error']),
                                  axis=1, result_type='expand').rename(
                columns={0: 'p1_a1_mas', 1: 'p1_omega_deg', 2: 'p1_OMEGA_deg', 3: 'p1_incl_deg',
                         4: 'sigma_p1_a1_mas', 5: 'sigma_p1_omega_deg', 6: 'sigma_p1_OMEGA_deg', 7: 'sigma_p1_incl_deg'})
            tmp['p1_a1_div_sigma_a1_mas'] = tmp['p1_a1_mas']/tmp['sigma_p1_a1_mas']

            if 'period' in self._obj.columns:
                self._obj.loc[:, 'p1_period_day'] = self._obj.loc[:, 'period']
                self._obj['sigma_p1_period_day'] = self._obj['period_error']
            self._obj['sigma_p1_ecc'] = self._obj['eccentricity_error']

            self._obj = pd.concat([self._obj, tmp], axis='columns')
        else:
            print('Columns already exist!')
        return self._obj


    filled_solution_parameters = {'Orbital': ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes', 'b_thiele_innes',
                              'f_thiele_innes', 'g_thiele_innes', 'eccentricity', 'period', 't_periastron']}
    filled_solution_parameters['ExtrasolarPlanets'] = filled_solution_parameters['Orbital']
    filled_solution_parameters['AstroSpectroSB1'] = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes', 'b_thiele_innes',
                              'f_thiele_innes', 'g_thiele_innes', 'c_thiele_innes', 'h_thiele_innes',
                              'center_of_mass_velocity', 'eccentricity', 'period', 't_periastron']



    @staticmethod
    def get_indices_in_corr_vec(parameters=None):
        if parameters == ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']:
            parameter_index_in_bit_index = np.array([5, 6, 7, 8])
            n_params_total = 12
            parameter_index_not_in_bit_index = np.setdiff1d(np.arange(n_params_total),
                                                            parameter_index_in_bit_index)
            # iu1 = np.triu_indices(n_params_total, k=+1)
            iu1 = np.tril_indices(n_params_total, k=-1)
            a = np.zeros((n_params_total, n_params_total))
            a[iu1] = 1
            a[parameter_index_not_in_bit_index, :] = 0
            a[:, parameter_index_not_in_bit_index] = 0
            # indices_in_corr_vec = np.where(a[iu1].flatten())[0]
            indices_in_corr_vec = np.where(a[iu1])[0]

        return indices_in_corr_vec


    @staticmethod
    def draw_monte_carlo_solution_parameters(row, n_mc=1000, parameters=None, indices_in_corr_vec=None):
        """ Return dataframe with MC samples for an individual solution."""

        if parameters is None:
            # sample all parameters
            parameters = NssDataFrame.filled_solution_parameters[row['nss_solution_type']]

        # bin(df.loc[83, 'bit_index'])
        elif parameters == ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']:

            if 0:
                # determine index in corr_vec corresponding to ABFG
                parameter_index_in_bit_index = np.array([5, 6, 7, 8])
                n_params_total = 12
                parameter_index_not_in_bit_index = np.setdiff1d(np.arange(n_params_total),
                                                                parameter_index_in_bit_index)
                # iu1 = np.triu_indices(n_params_total, k=+1)
                iu1 = np.tril_indices(n_params_total, k=-1)
                a = np.zeros((n_params_total, n_params_total))
                a[iu1] = 1
                a[parameter_index_not_in_bit_index, :] = 0
                a[:, parameter_index_not_in_bit_index] = 0
                # indices_in_corr_vec = np.where(a[iu1].flatten())[0]
                indices_in_corr_vec = np.where(a[iu1])[0]

                # row['corr_vec'] = row['corr_vec'][indices_in_corr_vec]
                # logging.debug(index)
                # logging.debug(self._obj['corr_vec'].info())
                # logging.debug(self._obj.loc[index, 'corr_vec'][indices_in_corr_vec])

            row['corr_vec'] = row['corr_vec'][indices_in_corr_vec]


        cov_matrix = covariance_matrix(row, parameters=parameters)
        solution_parameters = row[parameters].astype(float)

        # draw random data accounting for covariances
        if np.isnan(cov_matrix).any(): # this corresponds to circular orbits
            return None
        # try:
        np.random.seed(row['source_id']%10000)
        data_mc = np.random.multivariate_normal(solution_parameters, cov_matrix, size=n_mc)

        df = pd.DataFrame(data_mc, columns=parameters)
        return df

    @staticmethod
    def compute_monte_carlo_resampled_quantiles(row, n_mc=1000, parameters=['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes'],
                                                indices_in_corr_vec=None):
        """Use row input to resample parameters according to covariance matrix and return quantiles.

        Parameters
        ----------
        row
        n_mc
        parameters

        Returns
        -------

        """

        mcdf = NssDataFrame.draw_monte_carlo_solution_parameters(row, n_mc, parameters, indices_in_corr_vec)
        if mcdf is None:
            return None
        mcdf = mcdf.nssmc.add_geometric_elements()
        # mcdf = mcdf.nssmc.add_companion_mass_estimate(m1_MS=m1_MS, sigma_m1_MS=sigma_m1_MS)
        mcdf_quantiles = mcdf.nssmc.get_quantiles()
        return mcdf_quantiles.iloc[0]



    # def sample_solution_parameters_monte_carlo2(self, index, n_mc=1000, parameters=None):
    #     """ Return dataframe with MC samples for an individual solution."""
    #     # logging.debug(f'Call to sample_solution_parameters_monte_carlo with index={index}')
    #
    #     row = self._obj.loc[index]
    #
    #     # bin(df.loc[83, 'bit_index'])
    #     if parameters == ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']:
    #
    #         # determine index in corr_vec corresponding to ABFG
    #         parameter_index_in_bit_index = np.array([5, 6, 7, 8])
    #         n_params_total = 12
    #         parameter_index_not_in_bit_index = np.setdiff1d(np.arange(n_params_total),
    #                                                         parameter_index_in_bit_index)
    #         # iu1 = np.triu_indices(n_params_total, k=+1)
    #         iu1 = np.tril_indices(n_params_total, k=-1)
    #         a = np.zeros((n_params_total, n_params_total))
    #         a[iu1] = 1
    #         a[parameter_index_not_in_bit_index, :] = 0
    #         a[:, parameter_index_not_in_bit_index] = 0
    #         # indices_in_corr_vec = np.where(a[iu1].flatten())[0]
    #         indices_in_corr_vec = np.where(a[iu1])[0]
    #
    #         # row['corr_vec'] = row['corr_vec'][indices_in_corr_vec]
    #         # logging.debug(index)
    #         # logging.debug(self._obj['corr_vec'].info())
    #         # logging.debug(self._obj.loc[index, 'corr_vec'][indices_in_corr_vec])
    #
    #         row['corr_vec'] = row['corr_vec'][indices_in_corr_vec]
    #
    #
    #     else:
    #
    #         if self._obj.loc[index, 'nss_solution_type'] in ['Orbital', 'ExtrasolarPlanets']:
    #             parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes', 'b_thiele_innes',
    #                           'f_thiele_innes', 'g_thiele_innes', 'eccentricity', 'period', 't_periastron']
    #         if self._obj.loc[index, 'nss_solution_type'] in ['AstroSpectroSB1']:
    #             parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes', 'b_thiele_innes',
    #                           'f_thiele_innes', 'g_thiele_innes', 'c_thiele_innes', 'h_thiele_innes',
    #                           'center_of_mass_velocity', 'eccentricity', 'period', 't_periastron']
    #
    #     #
    #     # # determine index in corr_vec corresponding to ABFG
    #     # parameter_index_in_bit_index = np.array([5, 6, 7, 8])
    #     # n_params_total = 12
    #     # parameter_index_not_in_bit_index = np.setdiff1d(np.arange(n_params_total), parameter_index_in_bit_index)
    #     # # iu1 = np.triu_indices(n_params_total, k=+1)
    #     # iu1 = np.tril_indices(n_params_total, k=-1)
    #     # a = np.zeros((n_params_total, n_params_total))
    #     # a[iu1] = 1
    #     # a[parameter_index_not_in_bit_index, :] = 0
    #     # a[:, parameter_index_not_in_bit_index] = 0
    #     # # indices_in_corr_vec = np.where(a[iu1].flatten())[0]
    #     # indices_in_corr_vec = np.where(a[iu1])[0]
    #     #
    #     # def compute_ge_mc(solution, n_mc=1000):
    #     #
    #     #     parameters = ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']
    #     #     if 1:
    #     #         # print(len(solution['corr_vec']))
    #     #         # print(solution['corr_vec'])
    #     #         # print(covariance_matrix(solution)[5:9, 5:9])
    #     #
    #     #         row = solution.copy()
    #     #         # print(row['corr_vec'])
    #     #         row['corr_vec'] = row['corr_vec'][indices_in_corr_vec]
    #     #             # print(row['corr_vec'])
    #     #         cov_matrix = covariance_matrix(row, parameters=parameters)
    #     #         # print(cov_matrix)
    #     #     else:
    #     cov_matrix = covariance_matrix(row, parameters=parameters)
    #     # solution_parameters = self._obj.loc[index, parameters].astype(np.float)
    #     solution_parameters = row[parameters].astype(np.float)
    #
    #     # draw random data accounting for covariances
    #     if np.isnan(cov_matrix).any(): # this corresponds to circular orbits
    #         return None
    #     # try:
    #     np.random.seed(index)
    #     data_mc = np.random.multivariate_normal(solution_parameters, cov_matrix, size=n_mc)
    #
    #     df = pd.DataFrame(data_mc, columns=parameters)
    #     return df


    def sample_solution_parameters_monte_carlo(self, index, n_mc=1000):
        """ Return dataframe with MC samples for an individual solution."""
        # logging.debug(f'Call to sample_solution_parameters_monte_carlo with index={index}')

        if ('Orbital' in self._obj.loc[index, 'nss_solution_type']) or (self._obj.loc[index, 'nss_solution_type'] in ['ExtrasolarPlanets']):
            parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes', 'b_thiele_innes',
                          'f_thiele_innes', 'g_thiele_innes', 'eccentricity', 'period', 't_periastron']
        if self._obj.loc[index, 'nss_solution_type'] in ['AstroSpectroSB1']:
            parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes', 'b_thiele_innes',
                          'f_thiele_innes', 'g_thiele_innes', 'c_thiele_innes', 'h_thiele_innes',
                          'center_of_mass_velocity', 'eccentricity', 'period', 't_periastron']

        #
        # # determine index in corr_vec corresponding to ABFG
        # parameter_index_in_bit_index = np.array([5, 6, 7, 8])
        # n_params_total = 12
        # parameter_index_not_in_bit_index = np.setdiff1d(np.arange(n_params_total), parameter_index_in_bit_index)
        # # iu1 = np.triu_indices(n_params_total, k=+1)
        # iu1 = np.tril_indices(n_params_total, k=-1)
        # a = np.zeros((n_params_total, n_params_total))
        # a[iu1] = 1
        # a[parameter_index_not_in_bit_index, :] = 0
        # a[:, parameter_index_not_in_bit_index] = 0
        # # indices_in_corr_vec = np.where(a[iu1].flatten())[0]
        # indices_in_corr_vec = np.where(a[iu1])[0]
        #
        # def compute_ge_mc(solution, n_mc=1000):
        #
        #     parameters = ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']
        #     if 1:
        #         # print(len(solution['corr_vec']))
        #         # print(solution['corr_vec'])
        #         # print(covariance_matrix(solution)[5:9, 5:9])
        #
        #         row = solution.copy()
        #         # print(row['corr_vec'])
        #         row['corr_vec'] = row['corr_vec'][indices_in_corr_vec]
        #             # print(row['corr_vec'])
        #         cov_matrix = covariance_matrix(row, parameters=parameters)
        #         # print(cov_matrix)
        #     else:
        cov_matrix = covariance_matrix(self._obj.loc[index], parameters=parameters)
        solution_parameters = self._obj.loc[index, parameters].astype(float)

        # draw random data accounting for covariances
        if np.isnan(cov_matrix).any(): # this corresponds to circular orbits
            return None
        # try:
        np.random.seed(index)
        data_mc = np.random.multivariate_normal(solution_parameters, cov_matrix, size=n_mc)

        df = pd.DataFrame(data_mc, columns=parameters)
        return df

    # def compute_monte_carlo_resampled_quantiles(self, index, n_mc=1000, parameters=['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']):
    #
    #     # logging.debug(f'compute_monte_carlo_resampled_quantiles called with index={index}')
    #
    #     mcdf = self.sample_solution_parameters_monte_carlo2(index, n_mc, parameters=parameters)
    #     mcdf = mcdf.nssmc.add_geometric_elements()
    #     # mcdf = mcdf.nssmc.add_companion_mass_estimate(m1_MS=m1_MS, sigma_m1_MS=sigma_m1_MS)
    #     mcdf_quantiles = mcdf.nssmc.get_quantiles()
    #     # logging.debug(mcdf_quantiles)
    #     # 1/0
    #     return mcdf_quantiles.iloc[0]



    def add_geometric_elements_uncertainty_monte_carlo(self):
        """ Add GE with uncertainties using Monte Carlo for error propagation."""

        parameters = ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']
        indices_in_corr_vec = NssDataFrame.get_indices_in_corr_vec(parameters)
        tmp = self._obj.apply(NssDataFrame.compute_monte_carlo_resampled_quantiles, axis=1,
                              indices_in_corr_vec=indices_in_corr_vec)
        self._obj = pd.concat([self._obj, tmp], axis='columns')
        self._obj['significance_mc'] = self._obj['p1_a1_mas_mc_0.5']/self._obj['p1_a1_mas_mc_sigma_mean']
        self._obj['cos(inclination)'] = np.cos(np.deg2rad(self._obj['p1_incl_deg']))
        self._obj['cos(inclination)_mc'] = np.cos(np.deg2rad(self._obj['p1_incl_deg_mc_0.5']))

        return self._obj

    def add_geometric_elements_uncertainty_monte_carlo_deprecated(self):
        """ Add GE with uncertainties using Monte Carlo for error propagation."""

        # determine index in corr_vec corresponding to ABFG
        parameter_index_in_bit_index = np.array([5, 6, 7, 8])
        n_params_total = 12
        parameter_index_not_in_bit_index = np.setdiff1d(np.arange(n_params_total), parameter_index_in_bit_index)
        # iu1 = np.triu_indices(n_params_total, k=+1)
        iu1 = np.tril_indices(n_params_total, k=-1)
        a = np.zeros((n_params_total, n_params_total))
        a[iu1] = 1
        a[parameter_index_not_in_bit_index, :] = 0
        a[:, parameter_index_not_in_bit_index] = 0
        # indices_in_corr_vec = np.where(a[iu1].flatten())[0]
        indices_in_corr_vec = np.where(a[iu1])[0]

        def compute_ge_mc(solution, n_mc=1000):
            # logging.info('compute_ge_mc called with source={}'.format(solution['source_id']))

            parameters = ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']
            if 1:
                # print(len(solution['corr_vec']))
                # print(solution['corr_vec'])
                # print(covariance_matrix(solution)[5:9, 5:9])

                row = solution.copy()
                # print(row['corr_vec'])
                row['corr_vec'] = row['corr_vec'][indices_in_corr_vec]
                    # print(row['corr_vec'])
                cov_matrix = covariance_matrix(row, parameters=parameters)
                # print(cov_matrix)
            else:
                cov_matrix = covariance_matrix(solution)[5:9, 5:9]
            thiele_innes_parameters = solution[parameters].astype(float)

            # draw random data accounting for covariances
            if np.isnan(cov_matrix).any(): # this corresponds to circular orbits
                return None
            # try:
            data_mc = np.random.multivariate_normal(thiele_innes_parameters, cov_matrix, size=n_mc)
            # except np.linalg.LinAlgError as e:
            #     # logging.info(cov_matrix)
            #     # logging.warn(f'compute_ge_mc: invalid covariance matrix: {e}')
            #     return None

            # compute geometric elements for every simulation
            geometric_parameters = geometric_elements(np.array(data_mc).T, post_process=True)
            ce_df = pd.DataFrame(geometric_parameters.T, columns=['p1_a1_mas_mc', 'p1_omega_deg_mc', 'p1_OMEGA_deg_mc', 'p1_incl_deg_mc'])

            # compute quantiles
            df1 = ce_df.quantile([0.16, 0.5, 0.84])

            # construct single_row dataframe with all quantiles
            df2 = df1.stack().swaplevel()
            df2.index = df2.index.map('{0[0]}_{0[1]}'.format)
            df3 = df2.to_frame().T
            return df3.iloc[0]


        if ('p1_a1_mas_mc_0.5' not in self._obj.columns):# and ('sigma_p1_a1_mas' not in self._obj.columns):
            tmp = self._obj.apply(lambda x: compute_ge_mc(x), axis=1)

            # tmp['sigma_p1_a1_mas_mc'] = tmp['p1_a1_mas_mc_0.84'] - tmp['p1_a1_mas_mc_0.5'] -
            # tmp['p1_a1_div_sigma_a1_mas_mc'] = tmp['p1_a1_mas_mc_0.5']/tmp['sigma_p1_a1_mas']
            self._obj['sigma_p1_period_day'] = self._obj['period_error']
            self._obj['sigma_p1_ecc'] = self._obj['eccentricity_error']

            self._obj = pd.concat([self._obj, tmp], axis='columns')
        else:
            print('Columns already exist!')
        return self._obj

    def plot_ti2ge_monte_carlo(self, index, n_mc=1000):
        """Generate figures illustrating the Monte Carlo transformation between ABFG and geometric elements."""

        try:
            import seaborn as sns
        except ImportError:
            logging.warn('Please install seaborn to use this method.')
            return
        if ('Orbital' in self._obj.loc[index, 'nss_solution_type']) or \
                (self._obj.loc[index, 'nss_solution_type'] in ['ExtrasolarPlanets']):
        # if self._obj.loc[index, 'nss_solution_type'] in ['Orbital', 'ExtrasolarPlanets']:
            parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes', 'b_thiele_innes',
                          'f_thiele_innes', 'g_thiele_innes', 'eccentricity', 'period', 't_periastron']
        if self._obj.loc[index, 'nss_solution_type'] in ['AstroSpectroSB1']:
            parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes', 'b_thiele_innes',
                          'f_thiele_innes', 'g_thiele_innes', 'c_thiele_innes', 'h_thiele_innes',
                          'center_of_mass_velocity', 'eccentricity', 'period', 't_periastron']

        cov_matrix = covariance_matrix(self._obj.loc[index], parameters=parameters)[5:9, 5:9]

        thiele_innes_parameters = self._obj.loc[
            index, ['a_thiele_innes', 'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes']].astype(
            float)
        np.random.seed(index)
        data_mc = np.random.multivariate_normal(thiele_innes_parameters, cov_matrix, size=n_mc)

        geometric_parameters = geometric_elements(np.array(data_mc).T, post_process=True)
        ce_df = pd.DataFrame(geometric_parameters.T,
                             columns=['p1_a1_mas_mc', 'p1_omega_deg_mc', 'p1_OMEGA_deg_mc',
                                      'p1_incl_deg_mc'])

        # # compute quantiles
        # df1 = ce_df.quantile([0.16, 0.5, 0.84])
        #
        # # construct single_row dataframe with all quantiles
        # df2 = df1.stack().swaplevel()
        # df2.index = df2.index.map('{0[0]}_{0[1]}'.format)
        # df3 = df2.to_frame().T

        self.add_geometric_elements_uncertainty_monte_carlo()

        df_mc = pd.DataFrame(data_mc, columns=['A', 'B', 'F', 'G'])

        g0 = sns.pairplot(df_mc, corner=True)
        #     pl.title(f'Input ABFG (N={n_mc})')
        pl.show()

        g = sns.pairplot(ce_df, corner=True)
        if 'p1_incl_deg' in self._obj.columns:
            for ii, key in enumerate(['p1_a1_mas', 'p1_omega_deg', 'p1_OMEGA_deg', 'p1_incl_deg']):
                g.axes[ii, ii].axvline(x=self._obj.loc[index, f'{key}'], ls='--', linewidth=2, c='k', label='Linear')
                g.axes[ii, ii].axvline(x=self._obj.loc[index, f'{key}_mc_0.5'], ls='-', linewidth=2, c='k', label='MC median')

            g.axes[0, 0].legend()

            # g.axes[3, 3].axvline(x=self._obj.loc[index, 'p1_incl_deg'], ls='--', linewidth=2, c='red',
            #                  label='nominal')
        # if 'p1_incl_deg_mc_0.5' in self._obj.columns:
        #     g.axes[3, 3].axvline(x=self._obj.loc[index, 'p1_incl_deg_mc_0.5'], ls='-', linewidth=2, c='blue',
        #                      label='MC median')
        pl.show()

        return g0, g

    def add_companion_mass_estimate(self, m1_MS=1., delta_mag=None):
        """Add m2_MJ column to dataframe."""

        if 'p1_a1_mas' not in self._obj.columns:
            self.add_geometric_elements()

        if m1_MS is None:
            assert 'm1_MS' in self._obj.columns
            logging.info('using column m1_MS values to compute companion mass')
            m1_MS = self._obj['m1_MS']

        a_m = pystrometry.convert_from_angular_to_linear(self._obj['p1_a1_mas'], self._obj['parallax'])
        m2_kg = pystrometry.pjGet_m2(m1_MS * pystrometry.MS_kg, a_m, self._obj['period'])
        self._obj['m2_MJ'] = m2_kg / pystrometry.MJ_kg
        self._obj['m2_MS'] = m2_kg / pystrometry.MS_kg
        self._obj['m1_MS'] = m1_MS

        return self._obj

    def add_absolute_magnitude(self, column='phot_g_mean_mag'):
        self._obj[f'absolute_{column}'] = self._obj[column] + 5 * np.log10(self._obj['parallax']) - 10
        return self._obj

    def add_colour(self):
        self._obj['bp_rp'] = self._obj['phot_bp_mean_mag'] - self._obj['phot_rp_mean_mag']
        return self._obj

    def plot_cmd(self, label=None, title=None, ax=None, show_cutoff=False, **kwargs):
        """Plot colour magnitude diagram."""

        if 'absolute_phot_g_mean_mag' not in self._obj.columns:
            self.add_absolute_magnitude()
        if 'bp_rp' not in self._obj.columns:
            self.add_colour()

        # fig = pl.figure()
        if ax is None:
            ax = pl.gca()

        self._obj.plot('bp_rp', 'absolute_phot_g_mean_mag', kind='scatter', ax=ax, label=label, **kwargs)  # , c='parallax'
        ax.set_title('{} ({} sources)'.format(title, len(self._obj)))
        ax.invert_yaxis()
        if show_cutoff:
            colour_cutoff = 3.5
            x = np.linspace(-1, colour_cutoff, 100)

            # def cutoff(x):
            #     return 5.5 + 3 * x

            def cutoff(x):
                return 6 + 3 * x
            pl.plot(x, cutoff(x), 'k-')
            pl.plot([colour_cutoff, colour_cutoff], [cutoff(colour_cutoff), 0], 'k-')

        ax.set_xlabel('$G_\mathrm{BP}-G_\mathrm{RP}$')
        ax.set_ylabel('$M_G$')

        # pl.show()
        # return ax

@pd.api.extensions.register_dataframe_accessor("nssmc")
class NssMonteCarloDataFrame(NssDataFrame):
    """Class to hold data used in Monte Carlo sampling of NSS solution."""

    def add_companion_mass_estimate(self, m1_MS=1., sigma_m1_MS=0, delta_mag=None):
        """Add m2_MJ column to dataframe."""

        if 'p1_a1_mas' not in self._obj.columns:
            self.add_geometric_elements()

        n_mc = len(self._obj)
        m1_MS = np.ones(n_mc) * m1_MS
        if sigma_m1_MS != 0:
            np.random.seed(1234)
            m1_MS += np.random.normal(0, sigma_m1_MS, n_mc)

        self._obj['m1_MS'] = m1_MS

        a_m = pystrometry.convert_from_angular_to_linear(self._obj['p1_a1_mas'], self._obj['parallax'])
        m2_kg = pystrometry.pjGet_m2(self._obj['m1_MS'] * pystrometry.MS_kg, a_m, self._obj['period'])
        self._obj['m2_MJ'] = m2_kg / pystrometry.MJ_kg
        self._obj['m2_MS'] = m2_kg / pystrometry.MS_kg

        return self._obj


    def get_quantiles(self, quantiles=[0.16, 0.5, 0.84]):
        # compute quantiles
        df1 = self._obj.quantile(quantiles)

        # construct single_row dataframe with all quantiles
        df2 = df1.stack().swaplevel()
        df2.index = df2.index.map('{0[0]}_mc_{0[1]}'.format)
        df3 = df2.to_frame().T

        keys = self._obj.columns
        for key in keys:
            df3[f'{key}_mc_sigma_upper'] = df3[f'{key}_mc_{quantiles[2]}'] - df3[f'{key}_mc_{quantiles[1]}']
            df3[f'{key}_mc_sigma_lower'] = df3[f'{key}_mc_{quantiles[1]}'] - df3[f'{key}_mc_{quantiles[0]}']
            df3[f'{key}_mc_sigma_mean'] = df3[[f'{key}_mc_sigma_upper', f'{key}_mc_sigma_lower']].mean(axis=1)
        return df3


def bfp_latex(x, key, prec=2):
    """Return formatted latex string showing median and upper/lower confidence interval"""
    value = x[f'{key}_mc_0.5']
    upper = x[f'{key}_mc_sigma_upper']
    lower = -1*x[f'{key}_mc_sigma_lower']
    s = f'{{{value:2.{prec}f}}}^{{{upper:+2.{prec}f}}}_{{{lower:+2.{prec}f}}}'
    return s

def parameter_with_error_latex(x, key, prec=2, suffix=None, prefix=None, include_dollar=True):
    """Return formatted latex string showing parameter with simple error"""
    value = x[f'{key}']
    if suffix is not None:
        uncertainty = x[f'{key}{suffix}']
    elif prefix is not None:
        uncertainty = x[f'{prefix}{key}']

    s = f'{{{value:2.{prec}f}}}\pm{{{uncertainty:2.{prec}f}}}'
    if include_dollar:
        s = '$'+s+'$'
    return s




def apply_elimination_cuts(table, selection_cuts, parameter_mapping):
    """Eliminate rows in astropy table based on input parameters.

    Parameters
    ----------
    table
    selection_cuts
    parameter_mapping

    Returns
    -------

    Examples
    --------

    selection_cuts = OrderedDict({'period_1': {'operator': '<', 'threshold': 1000.},
                  'period_2': {'operator': '>', 'threshold': 50.},
                  'm2sini': {'operator': '>', 'threshold': 10.},
                  })


    parameter_mapping = {'period': 'PER',
                         'ecc': 'ECC',
                         'm2sini': 'MSINI',
                         'omega': 'OM',
                         'plx': 'PAR',
                         }


    """
    string_repr = ''
    for field, parameters in selection_cuts.items():
        if parameters['operator'] == '>':
            remove_index = np.where(table[parameter_mapping[field]] > parameters['threshold'])[0]
        elif parameters['operator'] == '<':
            remove_index = np.where(table[parameter_mapping[field]] < parameters['threshold'])[0]
        table.remove_rows(remove_index)
        string_repr += '{:>10} {} {:>6}\n'.format(field, parameters['operator'],
                                                  parameters['threshold'])

    return table, string_repr


def apply_selection_cuts(table, selection_cuts, parameter_mapping):
    """

    Parameters
    ----------
    table
    selection_cuts
    parameter_mapping

    Returns
    -------

    """
    string_repr = ''
    for field, parameters in selection_cuts.items():
        field = field.split('_')[0]
        if parameters['operator'] == '>':
            remove_index = np.where(table[parameter_mapping[field]] < parameters['threshold'])[0]
        elif parameters['operator'] == '<':
            remove_index = np.where(table[parameter_mapping[field]] > parameters['threshold'])[0]
        table.remove_rows(remove_index)
        string_repr += '{:>10} {} {:>6}\n'.format(field, parameters['operator'],
                                                  parameters['threshold'])

    return table, string_repr


def period_phase_error(period_day_fit, period_day_truth, time_span_day):
    """Return the period phase error as defined in BH-011."""
    return np.abs((period_day_fit - period_day_truth)/period_day_truth * time_span_day/period_day_truth)


def make_comparison_figures(table, parameter_mapping, mapping_dr3id_to_starname,
                            highlight_index=None, description_str='',
                            save_plot=True, plot_dir=os.getcwd(), time_span_day=1000.,
                            period_phase_error_threshold=0.2):
    """

    Parameters
    ----------
    table
    parameter_mapping
    highlight_index
    description_str
    save_plot
    plot_dir

    Returns
    -------

    """

    # also save table with discrepancies
    discrepancy_table = Table()
    discrepancy_table['sourceId'] = table['sourceId']
    discrepancy_table['Name'] = table['Name']
    discrepancy_table['Name_dedreq'] = table['Name_dedreq-695']
    discrepancy_table['m2_mjup'] = table['{}_m2_mjup'.format('p1')]

    for miks_name, mapped_name in parameter_mapping.items():
        if miks_name not in 'plx'.split():
            miks_field = 'p1_{}'.format(miks_name)
        else:
            miks_field = '{}'.format(miks_name)
        if miks_field not in table.colnames:
            continue
        pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        pl.plot(table[mapped_name], table[miks_field], 'bo')

        discrepancy_table[miks_field] = table[miks_field]
        discrepancy_table[mapped_name] = table[mapped_name]
        discrepancy_table['{}_discrepancy'.format(miks_field)] = table[mapped_name] - table[miks_field]
        # discrepancy in percent
        # discrepancy_table['{}_discr_rel'.format(miks_field)] = 100.*np.abs(table[mapped_name] - table[miks_field])/np.abs(table[miks_field])
        discrepancy_table['{}_discr_rel'.format(miks_field)] = 100.*np.abs(table[mapped_name] - table[miks_field])/np.abs(table[mapped_name])

        if highlight_index is not None:
            pl.plot(table[mapped_name][highlight_index],
                    table[miks_field][highlight_index], 'ro', ms=15, mfc='none')
        pl.axis('equal')
        xymax = np.max(np.array([pl.xlim()[1], pl.ylim()[1]]))
        pl.plot([0, xymax], [0, xymax], 'k--')
        pl.xlabel('{} ({})'.format(mapped_name, table.meta['comparison_to']))
        pl.ylabel('{} (DU437)'.format(miks_field))
        pl.title('{} sources from {}'.format(len(table), table.meta['comparison_to']))
        pl.text(0.01, 0.99, description_str, horizontalalignment='left', verticalalignment='top',
                transform=pl.gca().transAxes)
        pl.show()
        if save_plot:
            figure_name = os.path.join(plot_dir, '{}_comparison_to_{}.pdf'.format(miks_field, table.meta['comparison_to']))
            pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)


    # period phase error:


    miks_name = 'period_day'
    miks_field = 'p1_{}'.format(miks_name)
    mapped_name = parameter_mapping[miks_name]

    period_day_fit = table[miks_field]
    period_day_truth = table[mapped_name]
    discrepancy_table['period_phase_error'] = period_phase_error(period_day_fit, period_day_truth, time_span_day)

    n_period_recovered = len(np.where(np.abs(discrepancy_table['period_phase_error'])<period_phase_error_threshold)[0])
    pl.figure(figsize=(8, 4), facecolor='w', edgecolor='k')
    pl.plot(period_day_truth, discrepancy_table['period_phase_error'], 'k.')
    pl.ylim((-1,1))
    pl.fill_between(pl.xlim(), period_phase_error_threshold, y2=-period_phase_error_threshold, color='g', alpha=0.5)
    pl.xlabel('Truth period (day)')
    pl.ylabel('Period phase error')
    description_str_2 = '{}/{} = {:2.1f}% within +/- {:2.1f}\n'.format(n_period_recovered, len(discrepancy_table), n_period_recovered/len(discrepancy_table)*100, period_phase_error_threshold)+description_str
    pl.text(0.01, 0.99, description_str_2, horizontalalignment='left', verticalalignment='top',
            transform=pl.gca().transAxes)
    pl.show()
    if save_plot:
        figure_name = os.path.join(plot_dir, 'period_phase_error_{}.pdf'.format(table.meta['comparison_to']))
        pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

    # pl.close('all')
    threshold = {'delta_chi2': {'value': 1000, 'operator': '>'},
                 'f_test_probability': {'value': 1e-100, 'operator': '<'}
                 }
    for miks_field in ['meritFunction', 'chi2WithPlanet', 'chi2SingleStar', 'delta_chi2', 'f_test_probability', 'p1_estSNratio', 'p1_period_snr']:
        pl.figure(figsize=(8, 4), facecolor='w', edgecolor='k')
        index = np.where(discrepancy_table['period_phase_error'] < 100)[0]
        pl.loglog(discrepancy_table['period_phase_error'][index], table[miks_field][index], 'bo', alpha=0.7)
        # pl.ylim((-1,1))
        # pl.fill_between(pl.xlim(), period_phase_error_threshold, y2=-period_phase_error_threshold, color='g', alpha=0.5)
        pl.xlabel('Period phase error')
        pl.ylabel(miks_field)
        n_passed_threshold = None
        if miks_field in ['delta_chi2', 'f_test_probability']:
            value = threshold[miks_field]['value']
            operator = threshold[miks_field]['operator']
            if operator == '>':
                n_passed_threshold = len(np.where(table[miks_field] > value)[0])
                pl.fill_between(pl.xlim(), value, y2=pl.ylim()[1], color='g', alpha=0.5)
            elif operator == '<':
                n_passed_threshold = len(np.where(table[miks_field] < value)[0])
                pl.fill_between(pl.xlim(), value, y2=pl.ylim()[0], color='g', alpha=0.5)

        pl.title('{} of {} systems shown. {} pass threshold'.format(len(index), len(table), n_passed_threshold))
        pl.text(0.01, 0.99, description_str, horizontalalignment='left', verticalalignment='top',
                transform=pl.gca().transAxes)
        pl.show()
        if save_plot:
            figure_name = os.path.join(plot_dir, 'period_phase_error_vs_{}_{}.pdf'.format(miks_field, table.meta['comparison_to']))
            pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)


    if 1:
        formats = {}

        for key in discrepancy_table.colnames:#['angular_distance', 'phot_g_mean_mag', 'parallax', 'pmra', 'pmdec']:
            if 'discr' in key:
                formats[key] = '%>2.1f'
            elif 'm2' in key:
                formats[key] = '%2.1f'
            # else:
            #     formats[key] = '%2.3f'
        discrepancy_table_file = os.path.join(plot_dir, 'comparison_to_{}.csv'.format(table.meta['comparison_to']))
        if 'p1_period_discr_rel' in discrepancy_table.colnames:
            discrepancy_table.sort('p1_period_discr_rel')
        discrepancy_table.write(discrepancy_table_file, format='ascii.fixed_width', delimiter=',',
                                bookend=False, overwrite=True, formats=formats)

    try:
        pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        # pl.plot(table['p1_a1'], np.abs(table['p1_period'] - table[parameter_mapping['period']]), 'bo')
        # pl.xlabel('Fitted semimajor axis (mas)')
        pl.plot(table['a1_mas_minimum'], np.abs(table['p1_period'] - table[parameter_mapping['period']]), 'bo')
        pl.xlabel('Expected semimajor axis (mas)')
        pl.ylabel('Period error (day)')
        pl.show()
        if save_plot:
            figure_name = os.path.join(plot_dir, 'period_error_vs_a1.pdf')
            pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)
    except KeyError:
        pass

    return discrepancy_table


def get_gaia_iad(source_id, t_ref_jd, epoch_data_dir, verbose=False):
    """Return Gaia Epoch Astrometry Data.

    Parameters
    ----------
    selected_systems
    index
    epoch_data_dir

    Returns
    -------

    """

    t_ref_mjd = Time(t_ref_jd, format='jd').mjd

    iad = gaia_astrometry.GaiaIad(source_id, epoch_data_dir)
    iad.load_data()

    iad_mjd = Time(iad.epoch_data[iad.time_column] * 365.25 + t_ref_jd, format='jd').mjd
    iad.epoch_data['MJD'] = iad_mjd

    iad.epoch_data_for_prototype = Table()
    iad.epoch_data_for_prototype['t-t_ref'] = iad.epoch_data[iad.time_column]

    for key in ['spsi_obs', 'cpsi_obs', 'ppfact_obs', 'da_mas_obs', 'errda_mas_obs', 'transitId',
                'direction_AL0_AC1', 'OB']:
        iad.epoch_data_for_prototype[key] = iad.epoch_data[key]
        if key in ['spsi_obs', 'cpsi_obs']:
            iad.epoch_data_for_prototype['t{}'.format(key)] = iad.epoch_data_for_prototype[
                                                                  't-t_ref'] * \
                                                              iad.epoch_data_for_prototype[key]

    iad.epoch_data = copy.deepcopy(iad.epoch_data_for_prototype)
    iad.time_column = 't-t_ref'
    iad.epoch_data['MJD'] = iad_mjd
    iad.t_ref_mjd = t_ref_mjd

    iad.scan_angle_definition = 'gaia'
    if verbose:
        iad.epoch_data.pprint()

    return iad


def make_orbit_system(selected_systems, index, scan_angle_definition, t_ref_mjd,
                      m1_MS=1., degenerate_orbit=False,
                      verbose=False):
    """Return an OrbitSystem for the specified input table row.

    Parameters
    ----------
    selected_systems
    index
    epoch_data_dir
    mapping_dr3id_to_starname
    plot_dir
    m1_MS
    rv
    show_plot
    degenerate_orbit

    Returns
    -------

    """


    alpha_mas = selected_systems['p1_a1_mas'][index]
    absolute_parallax_mas = selected_systems['plx_mas'][index]
    a_m = pystrometry.convert_from_angular_to_linear(alpha_mas, absolute_parallax_mas)
    P_day = selected_systems['p1_period_day'][index]
    m2_kg = pystrometry.pjGet_m2(m1_MS*pystrometry.MS_kg, a_m, P_day)
    m2_MJ = m2_kg/pystrometry.MJ_kg

    attribute_dict = {
        'offset_alphastar_mas': selected_systems['alphaStarOffset_mas'][index],
        'offset_delta_mas': selected_systems['deltaOffset_mas'][index],
        # 'RA_deg': 0.,
        # 'DE_deg': 0.,
        'RA_deg': selected_systems['alpha0_deg'][index],
        'DE_deg': selected_systems['delta0_deg'][index],
        # 'plx_mas': selected_systems['plx'][index],
        'absolute_plx_mas': selected_systems['plx_mas'][index],
        'muRA_mas': selected_systems['muAlphaStar_masPyr'][index],
        'muDE_mas': selected_systems['muDelta_masPyr'][index],
        'P_day': selected_systems['p1_period_day'][index],
        'ecc': selected_systems['p1_ecc'][index],
        'omega_deg': selected_systems['p1_omega_deg'][index],
        'OMEGA_deg': selected_systems['p1_OMEGA_deg'][index],
        'i_deg': selected_systems['p1_incl_deg'][index],
        'a_mas': selected_systems['p1_a1_mas'][index],
        'Tp_day': t_ref_mjd + selected_systems['p1_Tp_day-T0'][index],
        'm1_MS': m1_MS,
        'm2_MJ': m2_MJ,
        'Tref_MJD': t_ref_mjd,
        'scan_angle_definition': scan_angle_definition,

    }


    if degenerate_orbit:
        attribute_dict['omega_deg'] += 180.
        attribute_dict['OMEGA_deg'] += 180.

    orbit = pystrometry.OrbitSystem(attribute_dict=attribute_dict)
    if verbose:
        print(orbit)

    return orbit



def make_astrometric_orbit_plotter(selected_systems, index, epoch_data_dir, degenerate_orbit=False,
                                   verbose=False, m1_MS=1.):
    """Return AstrometricOrbitPlotter object

    Parameters
    ----------
    selected_systems
    index
    epoch_data_dir
    degenerate_orbit
    verbose
    m1_MS

    Returns
    -------

    """

    source_id = selected_systems['sourceId'][index]
    t_ref_jd = selected_systems['T0_JD'][index]

    iad = get_gaia_iad(source_id, t_ref_jd, epoch_data_dir, verbose=verbose)

    orbit = make_orbit_system(selected_systems, index, iad.scan_angle_definition, iad.t_ref_mjd, m1_MS=m1_MS,
                              degenerate_orbit=degenerate_orbit, verbose=verbose)

    # set coeffMatrix in orbit object
    ppm_signal_mas = orbit.ppm(iad.epoch_data['MJD'], psi_deg=np.rad2deg(
        np.arctan2(iad.epoch_data['spsi_obs'], iad.epoch_data['cpsi_obs'])),
                               offsetRA_mas=selected_systems['alphaStarOffset_mas'][index],
                               offsetDE_mas=selected_systems['deltaOffset_mas'][index],
                               externalParallaxFactors=iad.epoch_data['ppfact_obs'], verbose=True)
    # 1/0

    plot_dict = {}
    plot_dict['model_parameters'] = {0: orbit.attribute_dict}
    plot_dict['linear_coefficients'] = {'matrix': orbit.coeffMatrix}  # dict ('matrix', 'table')
    plot_dict['data_type'] = '1d'
    if hasattr(iad, 'xi'):
        plot_dict['data_type'] = 'gaia_2d'
    else:
        plot_dict['data_type'] = '1d'
    plot_dict['scan_angle_definition'] = iad.scan_angle_definition

    for key in iad.epoch_data.colnames:
        if '_obs' in key:
            new_key = key.replace('_obs', '')
            if new_key == 'errda_mas':
                new_key = 'sigma_da_mas'
            iad.epoch_data[new_key] = iad.epoch_data[key]

    plot_dict['data'] = iad

    if verbose:
        iad.epoch_data.pprint()

    axp = pystrometry.AstrometricOrbitPlotter(plot_dict)

    # axp.print_residual_statistics()
    return axp


def make_orbit_figures(selected_systems, index, epoch_data_dir, mapping_dr3id_to_starname=None,
                      plot_dir=os.path.expanduser('~'),
                      m1_MS=1., rv=None, show_plot=True, degenerate_orbit=False, verbose=False):


    axp = make_astrometric_orbit_plotter(selected_systems, index, epoch_data_dir,
                                         degenerate_orbit=degenerate_orbit, verbose=verbose, m1_MS=m1_MS)


    iad = axp.data

    n_curve = 1500
    timestamps_curve_2d = np.linspace(np.min(iad.epoch_data['MJD']), np.max(iad.epoch_data['MJD']), n_curve)
    axp.t_curve_MJD = timestamps_curve_2d

    if 'phot_g_mean_mag' in selected_systems.colnames:
        mag_str = ' $G$={:2.1f}'.format(selected_systems['phot_g_mean_mag'][index])
    else:
        mag_str = ''

    if mapping_dr3id_to_starname is not None:
        axp.title = 'Gaia DR3 {} ({}{})'.format(source_id, mapping_dr3id_to_starname[source_id], mag_str)
        name_seed = 'DR3_{}_{}'.format(source_id, mapping_dr3id_to_starname[source_id])
    else:
        name_seed = 'DR3_{}'.format(source_id)

    argument_dict = {'plot_dir': plot_dir, 'ppm_panel': True, 'frame_residual_panel': True,
             'orbit_only_panel': False, 'ppm_description': 'default', 'epoch_omc_description': 'default',
             'orbit_description': 'default', 'arrow_offset_x': +100, 'arrow_offset_y': +100,
             'name_seed': name_seed, 'scan_angle_definition': scan_angle_definition}

    argument_dict['save_plot'] = True
    argument_dict['omc_panel'] = True
    argument_dict['orbit_only_panel'] = False
    # argument_dict['make_condensed_summary_figure'] = True
    # argument_dict['make_xy_residual_figure'] = True
    argument_dict['make_condensed_summary_figure'] = False
    argument_dict['make_xy_residual_figure'] = False
    argument_dict['make_1d_overview_figure'] = True
    argument_dict['excess_noise'] = selected_systems['excessNoise'][index]
    argument_dict['merit_function'] = selected_systems['meritFunction'][index]

    if show_plot:
        axp.plot(argument_dict=argument_dict)

        if rv is not None:
            from ..pystrometry import plot_rv_data

            my_orbit = copy.deepcopy(orbit)
            # my_orbit.m2_MJ = orbit.m2_MJ/10.
            plot_rv_data(rv, orbit_system=my_orbit, n_orbit=np.ceil(np.ptp(rv['MJD'])/orbit.P_day)+1)
            pl.show()

    return axp


def make_orbit_figure(selected_systems, index, epoch_data_dir, mapping_dr3id_to_starname=None,
                      plot_dir=os.path.expanduser('~'),
                      m1_MS=1., rv=None, show_plot=True, degenerate_orbit=False, epoch_data_suffix=None):

    source_id = selected_systems['sourceId'][index]
    t_ref_jd = selected_systems['T0_JD'][index]
    t_ref_mjd = Time(t_ref_jd, format='jd').mjd

    if epoch_data_suffix is None:
        iad = gaia_astrometry.GaiaIad(source_id, epoch_data_dir)
    else:
        iad = gaia_astrometry.GaiaIad(source_id, epoch_data_dir, epoch_data_suffix=epoch_data_suffix)

    iad.load_data(filter_on_frame_uncertainty=True)

    # pl.close('all')
    # pl.figure()
    # pl.hist(iad.epoch_data['errda_mas_obs'])
    # # pl.show()
    # pl.savefig('test.png')



    iad_mjd = Time(iad.epoch_data[iad.time_column]*365.25+t_ref_jd, format='jd').mjd
    iad.epoch_data['MJD'] = iad_mjd

    iad.epoch_data_for_prototype = Table()
    iad.epoch_data_for_prototype['t-t_ref'] = iad.epoch_data[iad.time_column]

    for key in ['spsi_obs', 'cpsi_obs', 'ppfact_obs', 'da_mas_obs', 'errda_mas_obs', 'transitId',
                'direction_AL0_AC1', 'OB']:
        iad.epoch_data_for_prototype[key] = iad.epoch_data[key]
        if key in ['spsi_obs', 'cpsi_obs']:
            iad.epoch_data_for_prototype['t{}'.format(key)] = iad.epoch_data_for_prototype['t-t_ref'] \
                                                              * iad.epoch_data_for_prototype[key]

    iad.epoch_data = copy.deepcopy(iad.epoch_data_for_prototype)
    iad.time_column = 't-t_ref'
    iad.epoch_data['MJD'] = iad_mjd
    iad.t_ref_mjd = t_ref_mjd

    scan_angle_definition = 'gaia'

    # loop over every companion in system
    from collections import OrderedDict
    model_parameters = OrderedDict()
    orbit_description = OrderedDict()
    # for planet_index in np.arange(1, selected_systems['Nplanets'][index]+1):
    for planet_index in np.arange(selected_systems['Nplanets'][index]):
        planet_number = planet_index + 1
        alpha_mas = selected_systems['p{}_a1_mas'.format(planet_number)][index]
        absolute_parallax_mas = selected_systems['plx_mas'][index]
        a_m = pystrometry.convert_from_angular_to_linear(alpha_mas, absolute_parallax_mas)
        P_day = selected_systems['p{}_period_day'.format(planet_number)][index]
        m2_kg = pystrometry.pjGet_m2(m1_MS*pystrometry.MS_kg, a_m, P_day)
        m2_MJ = m2_kg/pystrometry.MJ_kg

        attribute_dict = {
            'offset_alphastar_mas': selected_systems['alphaStarOffset_mas'][index],
            'offset_delta_mas': selected_systems['deltaOffset_mas'][index],
            # 'RA_deg': 0.,
            # 'DE_deg': 0.,
            'RA_deg': selected_systems['alpha0_deg'][index],
            'DE_deg': selected_systems['delta0_deg'][index],
            # 'plx_mas': selected_systems['plx'][index],
            'absolute_plx_mas': selected_systems['plx_mas'][index],
            'muRA_mas': selected_systems['muAlphaStar_masPyr'][index],
            'muDE_mas': selected_systems['muDelta_masPyr'][index],
            'P_day': selected_systems['p{}_period_day'.format(planet_number)][index],
            'ecc': selected_systems['p{}_ecc'.format(planet_number)][index],
            'omega_deg': selected_systems['p{}_omega_deg'.format(planet_number)][index],
            'OMEGA_deg': selected_systems['p{}_OMEGA_deg'.format(planet_number)][index],
            'i_deg': selected_systems['p{}_incl_deg'.format(planet_number)][index],
            'a_mas': selected_systems['p{}_a1_mas'.format(planet_number)][index],
            'Tp_day': iad.t_ref_mjd + selected_systems['p{}_Tp_day-T0'.format(planet_number)][index],
            'm1_MS': m1_MS,
            'm2_MJ': m2_MJ,
            'Tref_MJD': iad.t_ref_mjd,
            'scan_angle_definition': scan_angle_definition,
        }


        if degenerate_orbit:
            attribute_dict['omega_deg'] += 180.
            attribute_dict['OMEGA_deg'] += 180.

        # print(attribute_dict)
        # print(pystrometry.geometric_elements(np.array([selected_systems['p1_{}'.format(key)][index] for key in 'A B F G'.split()])))
        # print(pystrometry.mean_anomaly(iad.t_ref_mjd, attribute_dict['Tp_day'], attribute_dict['P_day']))

        if planet_index == 0:
            orbit = pystrometry.OrbitSystem(attribute_dict=attribute_dict)
            # print(orbit)

            # set coeffMatrix in orbit object
            ppm_signal_mas = orbit.ppm(iad.epoch_data['MJD'], psi_deg=np.rad2deg(
                np.arctan2(iad.epoch_data['spsi_obs'], iad.epoch_data['cpsi_obs'])),
                                       offsetRA_mas=selected_systems['alphaStarOffset_mas'][index], offsetDE_mas=selected_systems['deltaOffset_mas'][index],
                                       externalParallaxFactors=iad.epoch_data['ppfact_obs'], verbose=True)

        model_parameters[planet_index] = attribute_dict

        # display additional info on orbit panel
        # if 'P1_sigma_a1_mas' in selected_systems.columns:
        # p1_a1_div_sigma_a1_mas

        # if ('sigma_p1_a1_mas' in selected_systems.columns) and (selected_systems['Nplanets'][index]==1):
        if ('sigma_p1_a1_mas' in selected_systems.columns) and (selected_systems['Nplanets'][index]==1):
            # temporary: only for single-companion solutions
            orbit_descr = '$\\alpha={0[p1_a1_mas]:2.{prec}f}\\pm{0[sigma_p1_a1_mas]:2.{prec}f}$ mas (ratio={0[p1_a1_div_sigma_a1_mas]:2.1f})\n'.format(dict(selected_systems[index]), prec=3)
            orbit_descr += '$P={0[p1_period_day]:2.{prec}f}\\pm{0[sigma_p1_period_day]:2.{prec}f}$ d\n'.format(dict(selected_systems[index]), prec=1)
            orbit_descr += '$e={0[p1_ecc]:2.{prec}f}\\pm{0[sigma_p1_ecc]:2.{prec}f}$\n'.format(dict(selected_systems[index]), prec=3)
            orbit_descr += '$i={0[p1_incl_deg]:2.{prec}f}$ deg\n'.format(dict(selected_systems[index]), prec=2)
            orbit_descr += '$\\omega={0[p1_omega_deg]:2.{prec}f}$ deg\n'.format(dict(selected_systems[index]), prec=2)
            orbit_descr += '$\\Omega={0[p1_OMEGA_deg]:2.{prec}f}$ deg\n'.format(dict(selected_systems[index]), prec=2)
            orbit_descr += '$M_1={0:2.{prec}f}$ Msun\n'.format(m1_MS, prec=2)
            orbit_descr += '$M_2={0:2.{prec}f}$ Mjup\n'.format(m2_MJ, prec=2)
        else:
            orbit_descr = 'default'
        orbit_description[planet_index] = orbit_descr

    plot_dict = {}
    # plot_dict['model_parameters'] = {0: attribute_dict}
    plot_dict['model_parameters'] = model_parameters
    plot_dict['linear_coefficients'] = {'matrix': orbit.coeffMatrix} # dict ('matrix', 'table')
    plot_dict['data_type'] = '1d'
    if hasattr(iad, 'xi'):
        plot_dict['data_type'] = 'gaia_2d'
    else:
        plot_dict['data_type'] = '1d'
    plot_dict['scan_angle_definition'] = scan_angle_definition

    for key in iad.epoch_data.colnames:
        if '_obs' in key:
            new_key = key.replace('_obs', '')
            if new_key == 'errda_mas':
                new_key = 'sigma_da_mas'
            iad.epoch_data[new_key] = iad.epoch_data[key]

    plot_dict['data'] = iad

    # iad.epoch_data.pprint()

    axp = pystrometry.AstrometricOrbitPlotter(plot_dict)
    axp.print_residual_statistics()

    n_curve = 1500
    timestamps_curve_2d = np.linspace(np.min(iad.epoch_data['MJD']), np.max(iad.epoch_data['MJD']), n_curve)
    axp.t_curve_MJD = timestamps_curve_2d

    if 'phot_g_mean_mag' in selected_systems.colnames:
        mag_str = ' $G$={:2.1f}'.format(selected_systems['phot_g_mean_mag'][index])
    else:
        mag_str = ''

    if mapping_dr3id_to_starname is not None:
        axp.title = 'Gaia DR3 {} ({}{})'.format(source_id, mapping_dr3id_to_starname[source_id], mag_str)
        name_seed = 'DR3_{}_{}'.format(source_id, mapping_dr3id_to_starname[source_id].replace('/', '-'))
    else:
        name_seed = 'DR3_{}'.format(source_id)

    argument_dict = {'plot_dir': plot_dir, 'ppm_panel': True, 'frame_residual_panel': True,
             'orbit_only_panel': True, 'ppm_description': 'default', 'epoch_omc_description': 'default',
             'orbit_description': orbit_description, 'arrow_offset_x': +100, 'arrow_offset_y': +100,
             'name_seed': name_seed, 'scan_angle_definition': scan_angle_definition}

    argument_dict['save_plot'] = True
    argument_dict['omc_panel'] = True
    argument_dict['orbit_only_panel'] = False
    # argument_dict['orbit_only_panel'] = True
    # argument_dict['make_condensed_summary_figure'] = True
    # argument_dict['make_xy_residual_figure'] = True
    argument_dict['make_condensed_summary_figure'] = False
    argument_dict['make_xy_residual_figure'] = False
    argument_dict['make_1d_overview_figure'] = True
    argument_dict['excess_noise'] = selected_systems['excessNoise_mas'][index]
    argument_dict['merit_function'] = selected_systems['meritFunction'][index]

    if show_plot:
        axp.plot(argument_dict=argument_dict)
        if rv is not None:
            from ..pystrometry import plot_rv_data
            my_orbit = copy.deepcopy(orbit)
            # my_orbit.m2_MJ = orbit.m2_MJ/10.
            plot_rv_data(rv, orbit_system=my_orbit, n_orbit=np.ceil(np.ptp(rv['MJD'])/orbit.P_day)+1)
            pl.show()
    return axp


def show_best_solution(file, out_dir):
    """Make figures showing the MIKS best solution.

    Parameters
    ----------
    file : str
        csv file containing solutions

    Returns
    -------

    """
    data = Table.read(file, format='csv', data_start=2)
    units_table = Table.read(file, format='csv', data_start=1)
    units = {key: str(value) for key, value in zip(units_table[0].colnames, [units_table[0][c] for c in units_table.colnames])}

    # apply cuts on data
    threshold = 1e7
    for colname in data.colnames:
        if colname != 'sourceId':
            if np.any(np.abs(data[colname]) > threshold):
                data.remove_rows(np.where(np.abs(data[colname]) > threshold)[0])

    print('Eliminated {} of {} rows with values > {:1.0e}'.format(len(units_table)-len(data), len(data), threshold))

    plot_columns_simple(data, os.path.join(out_dir, 'best_solution_simple_plots'),
                        name_seed='best_solution', units=units)


def geometric_elements_with_uncertainties(thiele_innes_parameters, thiele_innes_parameters_errors=None, correlation_matrix=None,
                                          post_process=False, return_angles_in_deg=True):
    """
    Return geometrical orbit elements a, omega, OMEGA, i. If errors are not given
    they are assumed to be 0 and correlation matrix is set to identity.
    Complement to the pystrometry.geometric_elements function that allows to 
    compute parameter uncertainties as well.

    Parameters
    ----------
    thiele_innes_parameters : array
        Array of Thiele Innes parameters [A,B,F,G] in milli-arcsecond
    thiele_innes_parameters_errors : array, optional
        Array of the errors of the Thiele Innes parameters [A,B,F,G] in milli-arcsecond
    correlation_matrix : (4, 4) array, optional
        Correlation matrix for the Thiele Innes parameters [A,B,F,G]

    Returns
    -------
    geometric_parameters : array
        Orbital elements [a_mas, omega_deg, OMEGA_deg, i_deg]
    geometric_parameters_errors : array
        Errors of the orbital elements [a_mas, omega_deg, OMEGA_deg, i_deg]

    """

    # Checks on the errors and correlation matrix
    if (thiele_innes_parameters_errors is None) and (correlation_matrix is None):
        # Define errors to 0 and correlation matrix to identity
        thiele_innes_parameters_errors = [0,0,0,0]
        correlation_matrix = np.identity(4)
    elif (thiele_innes_parameters_errors is not None) and (correlation_matrix is not None):
        # If both are given continue to the calculation
        pass
    else:
        # If either one of them is provided but not the other raise an error
        raise ValueError("thieles_innes_parameters_errors and correlation_matrix must be" \
                          "specified together.")
            

    # Define uncorrelated (value, uncertainty) pairs
    A_u = (thiele_innes_parameters[0], thiele_innes_parameters_errors[0])
    B_u = (thiele_innes_parameters[1], thiele_innes_parameters_errors[1])
    F_u = (thiele_innes_parameters[2], thiele_innes_parameters_errors[2])
    G_u = (thiele_innes_parameters[3], thiele_innes_parameters_errors[3])

    # Create correlated quantities
    A, B, F, G = correlated_values_norm([A_u, B_u, F_u, G_u], correlation_matrix)

    p = (A ** 2 + B ** 2 + G ** 2 + F ** 2) / 2.
    q = A * G - B * F

    a_mas = unp.sqrt(p + unp.sqrt(p ** 2 - q ** 2))
    i_rad = unp.arccos(q / (a_mas ** 2.))
    omega_rad = (unp.arctan2(B - F, A + G) + unp.arctan2(-B - F, A - G)) / 2.
    OMEGA_rad = (unp.arctan2(B - F, A + G) - unp.arctan2(-B - F, A - G)) / 2.

    if post_process:
        # convert angles to nominal ranges
        omega_rad, OMEGA_rad = pystrometry.adjust_omega_OMEGA(omega_rad, OMEGA_rad)

    if return_angles_in_deg:
        # Convert radians to degrees
        i_deg = i_rad * 180 / np.pi
        omega_deg = omega_rad * 180 / np.pi
        OMEGA_deg = OMEGA_rad * 180 / np.pi
    else:
        i_deg = i_rad
        omega_deg = omega_rad
        OMEGA_deg = OMEGA_rad

    # Extract nominal values and standard deviations
    geometric_parameters = np.array([unp.nominal_values(a_mas), 
                                     unp.nominal_values(omega_deg), 
                                     unp.nominal_values(OMEGA_deg), 
                                     unp.nominal_values(i_deg)])

    geometric_parameters_errors = np.array([unp.std_devs(a_mas), 
                                            unp.std_devs(omega_deg), 
                                            unp.std_devs(OMEGA_deg), 
                                            unp.std_devs(i_deg)])

    return geometric_parameters, geometric_parameters_errors


def correlation_matrix(corr_vec):
    """ 
    This function reads the corr_vec from the nss_two_body_orbit table
    and converts it to a numpy array with the full correlation matrix.
    
    Parameters
    ----------
    input_table : pandas Series or dict
        A single row from the nss_two_body_orbit table for the desired target.

    Returns
    -------
    corr_mat : ndarray
        Correlation matrix for the specified parameters.
    """

    size = int((1+np.sqrt(8*len(corr_vec)+1))/2)
    # logging.debug(f'correlation_matrix size is {size}')
    corr_mat = np.ones([size, size], dtype=float)

    # Fill in lower triangle with corr_vec
    corr_mat[np.tril_indices(size, k=-1)] = corr_vec
    # Fill in upper triangle with mirror from lower triangle
    upper_triangle = np.triu_indices(size, k=1)
    corr_mat[upper_triangle] = corr_mat.T[upper_triangle]

    return corr_mat


def covariance_matrix(input_table, parameters=None):
    """ 
    Creates the covariance matrix from the nss_two_body_orbit
    table. By default it works for the 12  'Orbital' parameter solution but it can
    easily be adjusted to other solutions by providing the relevant parameters
    (in the same order that they are listed in the Gaia documentation). 
    
    Parameters
    ----------
    input_table : pandas Series or dict
        A single row from the nss_two_body_orbit table for the desired target.
    parameters : array-like, optional
        List of parameters for the corresponding solution of the desired
        target. They have to be in the same order that they appear in the
        Gaia documentation.

    Returns
    -------
    cov_mat : ndarray
        Covariance matrix for the specified parameters.
    """

    corr_vec = input_table['corr_vec']

    if parameters is None:
        # Use the 'Orbital' solution by default
        parameters = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes',
                      'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes',
                      'eccentricity', 'period', 't_periastron']
    else:
        # Check that the number of parameters provided matches the length of corr_vec
        assert len(parameters) == int((1+np.sqrt(8*len(corr_vec)+1))/2), \
               'Number of parameters does not match with the expected length of corr_vec:\n' \
               'Should be {} but is {}'.format(int((1+np.sqrt(8*len(corr_vec)+1))/2), len(parameters))

    size = len(parameters)

    # First create correlation matrix
    corr_mat = correlation_matrix(corr_vec)
    
    # Copy of correlation matrix to construct covariance matrix
    covar_mat = corr_mat.copy()

    # Get uncertainties and apply variances to correlation matrix
    pars_err = [par+'_error' for par in parameters]
    error = np.array([input_table[err_col] for err_col in pars_err])
    err_mat = np.dot(error.reshape(size, 1), error.reshape(1, size))
    covar_mat = covar_mat * err_mat
    
    return covar_mat