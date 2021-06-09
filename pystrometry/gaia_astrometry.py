"""Module to support working with Gaia epoch astrometry data.

Authors
-------

    Johannes Sahlmann



"""

import os

from astropy.table import Table
import numpy as np


def robust_scatter_estimate(data, verbose=True):
    """Return the Robust Scatter Estimate (RSE) es defined for Gaia.

    Parameters
    ----------
    data : ndarray
        Array of numbers.

    Returns
    -------

    References
    ----------

        @article{Lindegren:2012aa,
        Author = {{Lindegren}, L. and {Lammers}, U. and {Hobbs}, D. and {O'Mullane}, W. and {Bastian}, U. and {Hern{\'a}ndez}, J.},
        Journal = {\aap},
        Month = feb,
        Pages = {A78},
        Title = {{The astrometric core solution for the Gaia mission. Overview of models, algorithms, and software implementation}},
        Volume = 538,
        Year = 2012}

    """
    data_masked = np.ma.masked_array(data, np.isnan(data))
    data_cleaned = data_masked.data[data_masked.mask == False]
    if verbose:
        if len(data_masked) != len(data_cleaned):
            print('WARNING: data contains NaNs.')

    if not data_cleaned.size:
        return 0

    rse_factor = 0.390152
    rse = rse_factor * np.ptp(np.percentile(data_cleaned, [10, 90]))
    if np.isnan(rse):
        rse = 0.
    return rse



class GaiaIad(object):
    """Class for Gaia Epoch Astrometric Data."""

    def __init__(self, source_id, data_dir, epoch_data_suffix='_OBSERVATION_DATA_ALL.csv'):
        self.source_id = source_id
        self.data_dir = data_dir
        self.time_column = 'T-T0_yr'
        self.epoch_data_suffix = epoch_data_suffix

    def load_data(self, verbose=0, write_modified_iad=0, use_modified_epoch_data=False,
                  remove_outliers=True, filter_on_frame_uncertainty=False):

        self.epoch_data_file = os.path.join(self.data_dir, '{}{}'.format(self.source_id, self.epoch_data_suffix))
        self.epoch_data = Table.read(self.epoch_data_file)

        if 'direction_AL0_AC1' in self.epoch_data.colnames:
            if 0:
                self.xi = np.where(self.epoch_data['direction_AL0_AC1'] == 0)[0]
                self.yi = np.where(self.epoch_data['direction_AL0_AC1'] == 1)[0]
            else:
                remove_index = np.where(self.epoch_data['direction_AL0_AC1'] == 1)[0]
                self.epoch_data.remove_rows(remove_index)

        self.n_filtered_frames = 0
        self.n_original_frames = len(self.epoch_data)
        if filter_on_frame_uncertainty:
            rse_filter_threshold = 10
            rse = robust_scatter_estimate(self.epoch_data['errda_mas_obs'])
            # print(rse)
            remove_index = np.where(np.abs(self.epoch_data['errda_mas_obs']) > rse_filter_threshold*rse)[0]
            self.n_filtered_frames += len(remove_index)
            print('RSE cut removed {} frames (of {})'.format(self.n_filtered_frames, self.n_original_frames))
            self.epoch_data.remove_rows(remove_index)

        # remove focal plane transit averages used by GA code
        if 'isOriginal' in self.epoch_data.colnames:
            remove_index = np.where(self.epoch_data['isOriginal'] == 0)[0]
            self.epoch_data.remove_rows(remove_index)

        if remove_outliers:
            # remove_index = np.where(self.epoch_data['filterIntFlag'] == 2)[0]
            remove_index = np.where(self.epoch_data['filterIntFlag'] != 0)[0]
            self.epoch_data.remove_rows(remove_index)

        # sort by time
        self.epoch_data.sort(self.time_column)
        if verbose:
            print('Loaded IAD of {}'.format(self.source_id))
            print(self.epoch_data.info)

        # identify focal plane transits
        unique_transit_ids = np.unique(self.epoch_data['transitId'])
        self.n_fov_transit = len(unique_transit_ids)
        self.transit_id_name = 'OB'
        self.epoch_data[self.transit_id_name] = np.ones(len(self.epoch_data)).astype(int)
        for ob_number, transit_id in enumerate(unique_transit_ids):
            t_index = np.where(self.epoch_data['transitId'] == transit_id)[0]
            self.epoch_data[self.transit_id_name][t_index] = ob_number + 1

    def set_epoch_median_subtraction(self):
        """Subtract the median of an epoch from AL/AC measurements and add to table."""
        self.epoch_data['da_mas_obs_epoch_median_subtracted'] = np.zeros(len(self.epoch_data))
        for epoch_number in np.unique(self.epoch_data[self.transit_id_name]):
            t_index = np.where(self.epoch_data[self.transit_id_name] == epoch_number)[0]
            for direction_index in np.unique(self.epoch_data['direction_AL0_AC1'][t_index]):
                d_index = t_index[np.where(self.epoch_data['direction_AL0_AC1'][t_index] == direction_index)[0]]
                self.epoch_data['da_mas_obs_epoch_median_subtracted'][d_index] = self.epoch_data['da_mas_obs'][d_index] \
                                                                                 - np.median(self.epoch_data['da_mas_obs'][d_index])

    def astrometric_signal_to_noise(self, amplitude_mas):
        """Return astrometric signal to noise as defined in Sahlmann et al. 2011

        Parameters
        ----------
        amplitude_mas

        Returns
        -------

        """
        median_uncertainty_mas = np.median(self.epoch_data['errda_mas_obs'])
        n_ccd_transits = len(self.epoch_data)
        astrometric_snr = amplitude_mas * np.sqrt(n_ccd_transits)/median_uncertainty_mas

        return astrometric_snr