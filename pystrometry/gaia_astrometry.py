"""Module to support working with Gaia epoch astrometry data.

Authors
-------

    Johannes Sahlmann



"""

import os
from astropy.table import Table




class GaiaIad(object):
    """Class for Gaia Epoch Astrometric Data."""

    def __init__(self, source_id, data_dir):
        self.source_id = source_id
        self.data_dir = data_dir

        self.time_column = 'T-T0_yr'

        # self.cat_entry = HipStar(hip_id, catalog_dir)

    def load_data(self, verbose=0, write_modified_iad=0, use_modified_epoch_data=False):

        self.epoch_data_file = os.path.join(self.data_dir, '{}_OBSERVATION_DATA.csv'.format(self.source_id))
        self.epoch_data = Table.read(self.epoch_data_file)

        # sort by time
        self.epoch_data.sort(self.time_column)
        if verbose:
            print('Loaded IAD of {}'.format(self.source_id))
            print(self.epoch_data.info)


        # self.header, self.epoch_data = load_hipparcos(self.hip_id, self.catalog_dir,
        #                                               writeModCatPermission=write_modified_iad,
        #                                               verbose=verbose,
        #                                               use_modified_epoch_data=use_modified_epoch_data)
        # self.sigma_abscissa = self.epoch_data['SRES']

    # def compute_abscissa(self, parameter_code=123, reference_epoch_year=None, permute=0,
    #                      permute_2d=0):
    #     #     def xjGetHIPabscissa(hd,NastrometricParams=,Permute=,Permute2D=)
    #     '''/*  DOCUMENT  xjGetHIPabscissa(hp)
    #         return HIPPARCOS abscissa, reconstructed from the available 5-7-9 parameter solution give in the catalogue, including the abscissa residuals
    #         NOTE JSA 20091012: modified such that the computed abscissa does not take into account all astrometric parameters
    #         NOTE JSA 20091023: added possibility to control the number of astrometric parameters take into account, default is all NONE
    #
    #         NastrometricParams = 12: consider positions and parallax
    #         NastrometricParams = 23: consider parallax and proper motions
    #         NastrometricParams = 2:  consider only parallax
    #         NastrometricParams = 123:  all astromeric parameters are considered
    #
    #         WRITTEN  J. Sahlmann ObsGe 09.10.2009
    #         MODIFIED J. Sahlmann ObsGe 12.11.2009 added random permutation possibility
    #         MODIFIED J. Sahlmann ObsGe 18.08.2011 added permutation option for 2D data
    #     */'''
    #
    #     if hasattr(self, 'epoch_data') is False:
    #         self.load_data()
    #
    #     isoln = self.cat_entry.isoln
    #     n_obs = self.header['NOB'][0]
    #     if isoln in [1, 5]:
    #         abscissa_array = np.zeros((5, n_obs))
    #     else:
    #         warnings.warn('Solution type is not [1,5], returning abscissa with 5-parameter model')
    #         abscissa_array = np.zeros((5, n_obs))
    #
    #     if n_obs != len(self.epoch_data['PARF']):
    #         1 / 0
    #
    #     self.cpsi = np.array(self.epoch_data['CPSI'])
    #     self.spsi = np.array(self.epoch_data['SPSI'])
    #
    #     # if reference_epoch_year is not None:
    #     #     self.epoch = np.array(self.epoch_data['EPOCH'] + 1991.25 - reference_epoch_year)
    #     # else:
    #     self.epoch = np.array(self.epoch_data['EPOCH'])
    #     self.tcpsi = self.epoch * self.cpsi
    #     self.tspsi = self.epoch * self.spsi
    #
    #     self.parallax_factor = np.array(self.epoch_data['PARF'])
    #
    #     if parameter_code is not None:
    #         abscissa_array[2, :] = self.parallax_factor * self.cat_entry.main_cat_data['par_mas'][0]
    #
    #     if parameter_code in [12, 123]:
    #         abscissa_array[0, :] = self.cpsi * self.cat_entry.main_cat_data['alpha_rad'][
    #             0] * rad2mas
    #         abscissa_array[1, :] = self.spsi * self.cat_entry.main_cat_data['delta_rad'][
    #             0] * rad2mas
    #
    #     if parameter_code in [23, 123]:
    #         abscissa_array[3, :] = self.tcpsi * self.cat_entry.main_cat_data['mualpha_mas'][0]
    #         abscissa_array[4, :] = self.tspsi * self.cat_entry.main_cat_data['mudelta_mas'][0]
    #
    #     self.abscissa = np.sum(abscissa_array, axis=0) + np.array(self.epoch_data['RES'])
    #
    #
    # def linearfit_abscissa(self):
    #     '''
    #         func xj5parFit2(gd,da_obs)
    #     /* DOCUMENT xj5parFit(hg,da)
    #        function to get Chi2 when fitting 5 astrometric parameter model to HIPPARCOS data
    #        done by matrix inversion, since equation is linear
    #     '''
    #
    #     if hasattr(self, 'sigma_abscissa') is False:
    #         self.load_data()
    #
    #     weights = 1. / np.array(self.sigma_abscissa) ** 2
    #
    #     # dependent variables
    #     M = np.mat(self.abscissa)
    #
    #     #       diagonal covariance matrix of dependent variables
    #     S = np.mat(np.diag(weights))
    #
    #     # matrix of independent variables
    #     C = np.mat(np.vstack((self.cpsi, self.spsi, self.parallax_factor, self.tcpsi, self.tspsi)))
    #
    #     # initialise object
    #     res = linearfit.LinearFit(M, S, C)
    #
    #     # do the fit
    #     res.fit()
    #
    #     print("\n\n======== Results linearfit =========")
    #     res.display_results(precision=10)
    #     res.display_correlations()
    #
    #     self.linearfit_result = res
    #
    # def plot_linearfit(self, save_plot=0, plot_dir='', name_seed=''):
    #
    #     # if hasattr(self, 'epoch') is False:
    #     #     self.compute_abscissa()
    #     # if hasattr(self, 'linearfit_result') is False:
    #     #     self.linearfit_abscissa()
    #
    #     fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k');
    #     pl.clf();
    #     pl.plot(self.epoch, self.linearfit_result.residuals, 'ko')
    #     pl.errorbar(self.epoch, self.linearfit_result.residuals, yerr=self.sigma_abscissa,
    #                 ecolor='k', fmt=None)
    #     pl.xlabel('Time (years)')
    #     pl.ylabel('O-C (mas)')
    #     pl.axhline(0, color='0.7', ls='--')
    #     pl.show()
    #
    #     if save_plot:
    #         figName = os.path.join(plot_dir,
    #                                '%s_%s_linearfit_residuals.pdf' % (self.hip_id, name_seed))
    #         pl.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0)

