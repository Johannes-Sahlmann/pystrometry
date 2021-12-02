"""Module to support working with Gaia epoch astrometry data.

Authors
-------

    Johannes Sahlmann



"""

from collections import OrderedDict
import logging
import os

from astropy.table import Table
from astropy.time import Time
import astropy.units as u
import numpy as np

from .pystrometry import OrbitSystem, convert_from_angular_to_linear, pjGet_m2, \
    AstrometricOrbitPlotter, MS_kg, MJ_kg, AstrometricPpmPlotter

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


class GaiaValIad:
    """Class for Gaia Epoch Astrometric Data."""

    _transit_id_field = 'transitid'
    _fov_transit_id_field = 'OB'
    _obs_uncertainty_field = 'errda_mas_obs'
    scan_angle_definition = 'gaia'

    def __init__(self, source_id, epoch_data):
        self.source_id = source_id
        self.epoch_data = epoch_data
        self.n_filtered_frames = 0
        self.n_original_frames = len(self.epoch_data)

        self.epoch_df = self.epoch_data.to_pandas()

        # sort by time
        self.epoch_data.sort(self._time_field)
        self.set_fov_transit_id_field()

    # def set_fov_transit_id_field(self):
    #     # identify focal plane transits
    #     unique_transit_ids = self.epoch_data.to_pandas()[self._transit_id_field].unique() # not sorted
    #
    #     self.n_fov_transit = len(unique_transit_ids)
    #     self.epoch_data[self._fov_transit_id_field] = np.ones(len(self.epoch_data)).astype(int)
    #     for ob_number, transit_id in enumerate(unique_transit_ids):
    #         t_index = np.where(self.epoch_data[self._transit_id_field] == transit_id)[0]
    #         self.epoch_data[self._fov_transit_id_field][t_index] = ob_number + 1

    def set_fov_transit_id_field(self, eliminate_transits_overlapping_in_time=True):
        """Identify focal plane transits and number them."""
        self.epoch_df[self._fov_transit_id_field] = self.epoch_df.groupby(self._transit_id_field,
                                                                          sort=False).ngroup()

        # check whether any of the transits overlap in time
        groups = self.epoch_df.groupby(self._fov_transit_id_field)
        min_max_time = groups.agg([min, max])[self._time_field]
        transits_overlapping_in_time = (min_max_time.shift(-1, fill_value=0)['min'] < min_max_time['max'])[:-1]
        if np.any(transits_overlapping_in_time):
            fov_transit_id_to_exclude = transits_overlapping_in_time[transits_overlapping_in_time == True].index
            logging.debug(f'These transit numbers are overlapping in time: {fov_transit_id_to_exclude}')
            if eliminate_transits_overlapping_in_time:
                self.epoch_df = self.epoch_df[self.epoch_df[self._fov_transit_id_field].isin(fov_transit_id_to_exclude) == False]

        self.epoch_data = Table.from_pandas(self.epoch_df)
        assert np.all(np.diff(self.epoch_data[self._fov_transit_id_field]) >= 0)

    def __str__(self):
        """Return string describing the instance."""
        return 'GaiaValIad for source_id {} with {} frames)'.format(self.source_id,
                                                                    len(self.epoch_data))

    @classmethod
    def from_dataframe(cls, df, source_id, df_source_id_field='source_id'):
        """Extract data from dataframe and return astropy table."""
        return cls(source_id,
                   Table.from_pandas(df.query('{}==@source_id'.format(df_source_id_field))))

    def filter_on_frame_uncertainty(self, rse_filter_threshold=10):

        rse = robust_scatter_estimate(self.epoch_data[self._obs_uncertainty_field])
        remove_index = \
        np.where(np.abs(self.epoch_data[self._obs_uncertainty_field]) > rse_filter_threshold * rse)[
            0]
        self.n_filtered_frames += len(remove_index)
        print('RSE cut removed {} frames (of {})'.format(len(remove_index),
                                                         self.n_original_frames))
        self.epoch_data.remove_rows(remove_index)

    def filter_on_epoch_uncertainty(self, filter_threshold=2.5):

        epoch_signal_rms = self.epoch_data.group_by('OB').groups.aggregate(np.std)['da_mas_obs']
        indx = np.where(epoch_signal_rms > filter_threshold * np.median(epoch_signal_rms))[0]

        # indx = np.where(epoch_signal_rms > filter_threshold * robust_scatter_estimate(epoch_signal_rms))[0]

        epochs_to_exclude = np.unique(self.epoch_data.group_by('OB').groups[indx]['OB'].data)
        print(epochs_to_exclude)
        for epoch_to_exclude in epochs_to_exclude:
            remove_index = np.where(self.epoch_data['OB'] == epoch_to_exclude)[0]
            # if remove_index:
            self.n_filtered_frames += len(remove_index)
            print('Filter on epoch uncertainty cut removed {} frames (of {})'.format(self.n_filtered_frames,
                                                             self.n_original_frames))
            self.epoch_data.remove_rows(remove_index)

    def filter_on_strip(self, strip_threshold=2):

        remove_index = np.where(self.epoch_data['strip'] <= strip_threshold)[0]
        self.n_filtered_frames += len(remove_index)
        print('Filter on strip>{} removed {} frames (of {})'.format(strip_threshold, len(remove_index),
                                                         self.n_original_frames))
        self.epoch_data.remove_rows(remove_index)
                

class GaiaLpcParquetIad(GaiaValIad):
    """Class for LPC Data from parquet dumper."""

    _time_field = 'elapsedNanoSecs'
#     _time_field = 'deltat' # in years 
    time_column = 't-t_ref' # in years 
    _transit_id_field = 'transitId'
    # _fov_transit_id_field = 'OB'
    _mjd_field = 'MJD'
    _obs_uncertainty_field = 'sigma_da_mas'
    # scan_angle_definition = 'gaia'
        
    
    def __init__(self, source_id, epoch_data):
        """Initialise object from pandas.DataFrame."""
#         super().__init__(source_id, epoch_data)
        self.source_id = source_id
        self.epoch_data = epoch_data
        self.n_filtered_frames = 0
        self.n_original_frames = len(self.epoch_data)
        
        self.epoch_df = self.epoch_data.to_pandas()
#         self.sort_epochs_by_time(time_column=self._time_field)
        self.verify_missing_values()
        self.set_custom_fields()
        


    def verify_missing_values(self, remove_dubious_transits=True):    
        for key in ['w', 'obmt', 'theta']:
            if self.epoch_df[key].isnull().any():
                logging.warn(f'Some of the {key} data are masked.')
                if key == 'w':
                    logging.warn(f'Removing {len(self.epoch_df[key].isnull())}/{len(self.epoch_df)} rows where {key} data are masked.')
                    self.epoch_df = self.epoch_df[self.epoch_df[key].isnull() == False]
        
        if len(self.epoch_df) == 0:
            raise ValueError('IAD do not contain valid AL measurements. (all null)')                            
        
        n_unique_times = self.epoch_df[self._time_field].nunique()
        n_rows_df = len(self.epoch_df)
        if n_unique_times != n_rows_df:
            logging.warn(f'Values in time field {self._time_field} are not unique. There are {n_rows_df-n_unique_times}/{n_rows_df} duplicates.')
            value_counts = self.epoch_df[self._time_field].value_counts()
            logging.debug('Rows with duplicated timestamps:')
            logging.debug('\n{}'.format(self.epoch_df[self.epoch_df[self._time_field].isin(value_counts[value_counts>1].index)][[self._time_field, self._transit_id_field, 'w', 'strip']]))
#             logging.debug(f'epoch_data {self._time_field}:\n {epoch_df[self._time_field]}')
            if remove_dubious_transits:
                dubious_transit_ids = self.epoch_df[self.epoch_df[self._time_field].isin(value_counts[value_counts>1].index)][self._transit_id_field].values
                logging.debug(f'Removing data for {len(dubious_transit_ids)} transitIds')
                self.epoch_df = self.epoch_df[self.epoch_df[self._transit_id_field].isin(dubious_transit_ids) == False]
                
    
#         epoch_df.sort_values(self._time_field, inplace=True)
        self.epoch_data = Table.from_pandas(self.epoch_df)
    
    def set_custom_fields(self):
        
        self.epoch_data['spsi_obs'] = np.sin(self.epoch_data['theta'])
        self.epoch_data['cpsi_obs'] = np.cos(self.epoch_data['theta'])
        self.epoch_data['ppfact_obs'] = self.epoch_data['varpiFactorAl']
        self.epoch_data['da_mas'] = self.epoch_data['w']
        self.epoch_data['sigma_da_mas'] = self.epoch_data['wError']

        al_data = self.epoch_data['w'].data
        if (hasattr(al_data, 'mask')) and (np.all(al_data.mask)):
            raise ValueError('IAD do not contain valid AL measurements. (all masked)')

        self.epoch_df = self.epoch_data.to_pandas()

    def verify_timing(self):
#         epoch_df = self.epoch_data.to_pandas()
#         logging.debug(epoch_df[epoch_df[self._mjd_field].diff() <= 0])
#         logging.debug((epoch_df[self._mjd_field].diff() > 0).all())
#         assert (self.epoch_df[self._mjd_field].diff()[1:] >= 0).all()
#         assert (self.epoch_df[self._time_field].diff()[1:] > 0).all()
        assert (self.epoch_df[self._fov_transit_id_field].diff()[1:] >= 0).all()
            
        if np.any(np.diff(self.epoch_data['OB']) < 0):
            logging.debug(f'_fov_transit_id_field {self._fov_transit_id_field} is not monotonically increasing with time. Fixing it.')

    def sort_epochs_by_time(self, time_column=None, remove_dubious_times=True):   
        if time_column is None:
            time_column = self._mjd_field
            
        n_unique_times = self.epoch_df[time_column].nunique()
        n_rows_df = len(self.epoch_df)
        if n_unique_times != n_rows_df:
            logging.warn(f'Values in time field {time_column} are not unique. There are {n_rows_df-n_unique_times}/{n_rows_df} duplicates.')
            value_counts = self.epoch_df[time_column].value_counts()
            logging.debug('Rows with duplicated timestamps:')
            logging.debug('\n{}'.format(self.epoch_df[self.epoch_df[time_column].isin(value_counts[value_counts>1].index)][[time_column, self._time_field, self._transit_id_field, 'w', 'strip']]))
            if remove_dubious_times:
                dubious_times = self.epoch_df[self.epoch_df[time_column].isin(value_counts[value_counts>1].index)][time_column].values
                logging.debug(f'Removing data for {len(dubious_times)} times')
                self.epoch_df = self.epoch_df[self.epoch_df[time_column].isin(dubious_times) == False]
            self.epoch_data = Table.from_pandas(self.epoch_df)
            
        self.epoch_data.sort(time_column)
        self.epoch_df = self.epoch_data.to_pandas()
        assert np.all(self.epoch_data[time_column].data == self.epoch_df[time_column].values)

            
    def set_reference_time(self, reference_time):
        """Set reference time.

        Parameters
        ----------
        reference_time : astropy.time
            Reference time used in calculations.

        """

        self.t_ref_mjd = reference_time.mjd
        self.reference_time = reference_time

        #  convert to absolute time in MJD
        if 0:
            from agisp.utils import gaiaparameters
            tcb = gaiaparameters.obmt_to_tcb_approximation(self.epoch_data['obmt'] * gaiaparameters.obmt)
            self.epoch_data['MJD'] = Time(tcb, format='tcb_ns_2010', scale='tcb').mjd
            assert len(np.unique(self.epoch_data['MJD'])) == len(np.unique(self.epoch_data['obmt']))
        else:
#             self.epoch_data['MJD'] = Time(self.epoch_data[self._time_field] * 365.25 + reference_time.jd, format='jd').mjd
#             self.epoch_data['MJD'] = Time(self.epoch_data[self._time_field] + reference_time.jd, format='jd').mjd

            # public static final double REF_EPOCH_YR = 2010.0;
            reference_time = Time(2010.0, format='jyear')
            self.epoch_data['MJD'] = Time(self.epoch_data[self._time_field] * u.nanosecond.to(u.day) + reference_time.jd, format='jd').mjd
            self.epoch_df = self.epoch_data.to_pandas()
        
#         self.set_fov_transit_id_field()     
#         self.sort_epochs_by_time()

        self.sort_epochs_by_time(time_column=self._mjd_field)
#         self.sort_epochs_by_time(time_column=self._time_field)
        
        self.set_fov_transit_id_field()     
        
        self.epoch_data[self.time_column] = (self.epoch_data[self._mjd_field] - self.t_ref_mjd)/365.25        
        self.epoch_df = self.epoch_data.to_pandas()
        
        self.verify_timing()
        
class GaiaValIadHybrid(GaiaValIad):
    """Class for hybrid IAD that is produced internally in DU437 from a combination of StarObject and ObsConstStar data."""

    _time_field = 't_min_t0_yr'

    def set_reference_time(self, reference_time):
        """Set reference time.

        Parameters
        ----------
        reference_time : astropy.time
            Reference time used in calculations.

        """
        #  convert to absolute time in MJD
        self.epoch_data['MJD'] = Time(self.epoch_data[self._time_field] * 365.25 + reference_time.jd, format='jd').mjd

        self.epoch_data['t-t_ref'] = self.epoch_data[self._time_field]
        self.time_column = 't-t_ref'
        self.t_ref_mjd = reference_time.mjd
        self.reference_time = reference_time


def plot_individual_orbit(parameter_dict, iad, mapping_dr3id_to_starname=None,
                          plot_dir=os.path.expanduser('~'), m1_MS=1., degenerate_orbit=False):
    """Plot the orbit(s) of a single source due to orbiting companion(s).

    The code is prepared to handle multiple Keplerian solutions.

    Parameters
    ----------
    parameter_dict : dict
        Dictionary containing the orbit parameters
    iad : GaiaIad object
        Object holding the epoch astrometry data
    mapping_dr3id_to_starname : dict
        Mapping of source_id to common source name
    plot_dir : str
        Where to save the plot
    m1_MS : float
        Primary mass to use for companion mass estimation, under the assumption that the companion is dark.
    degenerate_orbit : bool
        Whether the orbit is 'degenerate'. This corresponds to the `other` set of geometric elements
        that produce the same astrometric orbit.

    Returns
    -------
    axp : AstrometricOrbitPlotter instance

    """
    model_parameters = OrderedDict()
    orbit_description = OrderedDict()

    # loop over every companion in system
    for planet_index in np.arange(parameter_dict['Nplanets']):
        planet_number = planet_index + 1
        alpha_mas = parameter_dict['p{}_a1_mas'.format(planet_number)]
        absolute_parallax_mas = parameter_dict['plx_mas']
        a_m = convert_from_angular_to_linear(alpha_mas, absolute_parallax_mas)
        P_day = parameter_dict['p{}_period_day'.format(planet_number)]
        m2_kg = pjGet_m2(m1_MS * MS_kg, a_m, P_day)
        m2_MJ = m2_kg / MJ_kg

        # dictionary
        attribute_dict = {
            'offset_alphastar_mas': parameter_dict['alphaStarOffset_mas'],
            'offset_delta_mas': parameter_dict['deltaOffset_mas'],
            'RA_deg': parameter_dict['alpha0_deg'],
            'DE_deg': parameter_dict['delta0_deg'],
            'absolute_plx_mas': parameter_dict['plx_mas'],
            'muRA_mas': parameter_dict['muAlphaStar_masPyr'],
            'muDE_mas': parameter_dict['muDelta_masPyr'],
            'P_day': parameter_dict['p{}_period_day'.format(planet_number)],
            'ecc': parameter_dict['p{}_ecc'.format(planet_number)],
            'omega_deg': parameter_dict['p{}_omega_deg'.format(planet_number)],
            'OMEGA_deg': parameter_dict['p{}_OMEGA_deg'.format(planet_number)],
            'i_deg': parameter_dict['p{}_incl_deg'.format(planet_number)],
            'a_mas': parameter_dict['p{}_a1_mas'.format(planet_number)],
            'Tp_day': iad.t_ref_mjd + parameter_dict['p{}_Tp_day-T0'.format(planet_number)],
            'm1_MS': m1_MS,
            'm2_MJ': m2_MJ,
            'Tref_MJD': iad.t_ref_mjd,
            'scan_angle_definition': iad.scan_angle_definition,
            'solution_type': parameter_dict['nss_solution_type']
        }

        if degenerate_orbit:
            attribute_dict['omega_deg'] += 180.
            attribute_dict['OMEGA_deg'] += 180.

        if planet_index == 0:
            orbit = OrbitSystem(attribute_dict=attribute_dict)

            # set coeffMatrix in orbit object
            ppm_signal_mas = orbit.ppm(iad.epoch_data['MJD'], psi_deg=np.rad2deg(
                np.arctan2(iad.epoch_data['spsi_obs'], iad.epoch_data['cpsi_obs'])),
                                       offsetRA_mas=parameter_dict['alphaStarOffset_mas'],
                                       offsetDE_mas=parameter_dict['deltaOffset_mas'],
                                       externalParallaxFactors=iad.epoch_data['ppfact_obs'],
                                       verbose=True)

        model_parameters[planet_index] = attribute_dict

        if 'varpiError' in parameter_dict.keys():
            ppm_description = ''
            ppm_description += '$\\varpi={0[varpi]:2.{prec}f}\\pm{0[varpiError]:2.{prec}f}$ mas\n'.format(parameter_dict, prec=2)
            ppm_description += '$\mu_\\mathrm{{ra^\\star}}={0[pmra]:2.{prec}f}\\pm{0[pmraError]:2.{prec}f}$ mas\n'.format(parameter_dict, prec=2)
            ppm_description += '$\mu_\\mathrm{{dec}}={0[pmdec]:2.{prec}f}\\pm{0[pmdecError]:2.{prec}f}$ mas\n'.format(parameter_dict, prec=2)
        else:
            ppm_description = 'default'

        if ('sigma_p1_a1_mas' in parameter_dict.keys()) and (parameter_dict['Nplanets'] == 1):
            # temporary: only for single-companion solutions
            # orbit_descr = '$\\alpha={0[p1_a1_mas]:2.{prec}f}\\pm{0[sigma_p1_a1_mas]:2.{prec}f}$ mas (ratio={0[p1_a1_div_sigma_a1_mas]:2.1f})\n'.format(
            #     dict(parameter_dict), prec=3)
            # orbit_descr += '$P={0[p1_period_day]:2.{prec}f}\\pm{0[sigma_p1_period_day]:2.{prec}f}$ d\n'.format(
            #     dict(parameter_dict), prec=1)
            # orbit_descr += '$e={0[p1_ecc]:2.{prec}f}\\pm{0[sigma_p1_ecc]:2.{prec}f}$\n'.format(
            #     dict(parameter_dict), prec=3)
            # orbit_descr += '$i={0[p1_incl_deg]:2.{prec}f}$ deg\n'.format(dict(parameter_dict),
            #                                                              prec=2)
            # orbit_descr += '$\\omega={0[p1_omega_deg]:2.{prec}f}$ deg\n'.format(
            #     dict(parameter_dict), prec=2)
            # orbit_descr += '$\\Omega={0[p1_OMEGA_deg]:2.{prec}f}$ deg\n'.format(
            #     dict(parameter_dict), prec=2)
            orbit_descr = '$\\alpha={0[p1_a1_mas]:2.{prec}f}\\pm{0[sigma_p1_a1_mas]:2.{prec}f}$ mas (ratio={0[p1_a1_div_sigma_a1_mas]:2.1f})\n'.format(dict(parameter_dict), prec=3)
            orbit_descr += '$P={0[p1_period_day]:2.{prec}f}\\pm{0[sigma_p1_period_day]:2.{prec}f}$ d\n'.format(dict(parameter_dict), prec=1)
            orbit_descr += '$e={0[p1_ecc]:2.{prec}f}\\pm{0[sigma_p1_ecc]:2.{prec}f}$\n'.format(dict(parameter_dict), prec=3)
            orbit_descr += '$i={0[p1_incl_deg]:2.{prec}f}\\pm{0[sigma_p1_incl_deg]:2.{prec}f}$ deg\n'.format(dict(parameter_dict), prec=2)
            orbit_descr += '$\\omega={0[p1_omega_deg]:2.{prec}f}\\pm{0[sigma_p1_omega_deg]:2.{prec}f}$ deg\n'.format(dict(parameter_dict), prec=2)
            orbit_descr += '$\\Omega={0[p1_OMEGA_deg]:2.{prec}f}\\pm{0[sigma_p1_OMEGA_deg]:2.{prec}f}$ deg\n'.format(dict(parameter_dict), prec=2)
            orbit_descr += '$M_1={0:2.{prec}f}$ Msun\n'.format(m1_MS, prec=2)
            orbit_descr += '$M_2={0:2.{prec}f}$ Mjup\n'.format(m2_MJ, prec=2)

        else:
            orbit_descr = 'default'
        orbit_description[planet_index] = orbit_descr

    plot_dict = {}
    plot_dict['model_parameters'] = model_parameters
    plot_dict['linear_coefficients'] = {'matrix': orbit.coeffMatrix}
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

    axp = AstrometricOrbitPlotter(plot_dict)

    n_curve = 1500
    timestamps_curve_2d = np.linspace(np.min(iad.epoch_data['MJD']), np.max(iad.epoch_data['MJD']),
                                      n_curve)
    axp.t_curve_MJD = timestamps_curve_2d

    if 'phot_g_mean_mag' in parameter_dict.keys():
        mag_str = '$G$={:2.1f}'.format(parameter_dict['phot_g_mean_mag'])
    else:
        mag_str = ''

    if mapping_dr3id_to_starname is not None:
        axp.title = 'Gaia DR3 {} ({} {})'.format(iad.source_id, mapping_dr3id_to_starname[source_id],
                                                mag_str)
        name_seed = 'DR3_{}_{}'.format(iad.source_id,
                                       mapping_dr3id_to_starname[iad.source_id].replace('/', '-'))
    else:
        axp.title = 'Gaia DR3 {} ({})'.format(iad.source_id, mag_str)
        name_seed = 'DR3_{}'.format(iad.source_id)

    argument_dict = {'plot_dir'             : plot_dir, 'ppm_panel': True,
                     'frame_residual_panel' : True, 'orbit_only_panel': True,
                     'ppm_description'      : ppm_description,
                     'epoch_omc_description': 'default',
                     'orbit_description'    : orbit_description, 'arrow_offset_x': +100,
                     'arrow_offset_y'       : +100, 'name_seed': name_seed,
                     'scan_angle_definition': iad.scan_angle_definition}

    argument_dict['save_plot'] = True
    argument_dict['omc_panel'] = True
    argument_dict['orbit_only_panel'] = False
    argument_dict['make_condensed_summary_figure'] = False
    argument_dict['make_xy_residual_figure'] = False
    argument_dict['make_1d_overview_figure'] = True
    argument_dict['excess_noise'] = parameter_dict['excessNoise_mas']
    argument_dict['merit_function'] = parameter_dict['meritFunction']

    axp.plot(argument_dict=argument_dict)

    return axp


def plot_individual_ppm(parameter_dict, iad, plot_dir=os.path.expanduser('~')):

    model_parameters = OrderedDict()
    orbit_description = OrderedDict()

    for planet_index in [0]:

        attribute_dict = {
            'offset_alphastar_mas': parameter_dict['alphaStarOffset_mas'],
            'offset_delta_mas': parameter_dict['deltaOffset_mas'],
            'RA_deg': parameter_dict['ra'],
            'DE_deg': parameter_dict['dec'],
            'absolute_plx_mas': parameter_dict['varpi'],
            'muRA_mas': parameter_dict['pmra'],
            'muDE_mas': parameter_dict['pmdec'],            
            'm2_MJ': 0.,
            'Tref_MJD': iad.t_ref_mjd,
            'scan_angle_definition': iad.scan_angle_definition,
        }
        attribute_dict['solution_type'] = parameter_dict['nss_solution_type']
            
    
        orbit = OrbitSystem(attribute_dict=attribute_dict)

        # set coeffMatrix in orbit object
        ppm_signal_mas = orbit.ppm(iad.epoch_data['MJD'], psi_deg=np.rad2deg(
            np.arctan2(iad.epoch_data['spsi_obs'], iad.epoch_data['cpsi_obs'])),
                                   offsetRA_mas=parameter_dict['alphaStarOffset_mas'], offsetDE_mas=parameter_dict['deltaOffset_mas'],
                                   externalParallaxFactors=iad.epoch_data['ppfact_obs'], verbose=True)

        model_parameters[planet_index] = attribute_dict

        orbit_descr = '{}  '.format('LPC offsets')
        orbit_descr += '\nScan angle spread = {:2.1f} deg'.format(np.ptp(np.rad2deg(iad.epoch_data['theta'])))

        ppm_description = ''
        ppm_description += '$\\varpi={0[varpi]:2.{prec}f}\\pm{0[varpiError]:2.{prec}f}$ mas\n'.format(parameter_dict, prec=2)
        ppm_description += '$\mu_\\mathrm{{ra^\\star}}={0[pmra]:2.{prec}f}\\pm{0[pmraError]:2.{prec}f}$ mas\n'.format(parameter_dict, prec=2)
        ppm_description += '$\mu_\\mathrm{{dec}}={0[pmdec]:2.{prec}f}\\pm{0[pmdecError]:2.{prec}f}$ mas\n'.format(parameter_dict, prec=2)

        orbit_description[planet_index] = orbit_descr

    plot_dict = {}
    plot_dict['model_parameters'] = model_parameters
    plot_dict['linear_coefficients'] = {'matrix': orbit.coeffMatrix} 
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


    axp = AstrometricPpmPlotter(plot_dict)
    axp.set_residuals()

    n_curve = 1500
    timestamps_curve_2d = np.linspace(np.min(iad.epoch_data['MJD']), np.max(iad.epoch_data['MJD']), n_curve)
    axp.t_curve_MJD = timestamps_curve_2d

    if 'GMag' in parameter_dict.keys():
        mag_str = ' $G$={:2.1f}'.format(parameter_dict['GMag'])
    else:
        mag_str = ''

    source_id = parameter_dict['sourceId']    
    axp.title = 'Gaia DR4 {} ({}, {})'.format(source_id, 'LPC', mag_str)
    name_seed = 'DR4_{}'.format(source_id)

    argument_dict = {'plot_dir': plot_dir, 'ppm_panel': True, 'frame_residual_panel': True,
             'ppm_description': ppm_description, 'epoch_omc_description': 'default',
             'orbit_description': orbit_description,
             'name_seed': name_seed, 'scan_angle_definition': iad.scan_angle_definition}

    argument_dict['save_plot'] = True
    argument_dict['omc_panel'] = True
    argument_dict['orbit_only_panel'] = False
    argument_dict['orbit_timeseries_panel'] = False
    argument_dict['make_condensed_summary_figure'] = False
    argument_dict['make_xy_residual_figure'] = False
    argument_dict['make_1d_overview_figure'] = True
    argument_dict['arrow_offset_x'] = parameter_dict['pmra']
    argument_dict['arrow_offset_x'] = parameter_dict['pmdec']
    
    argument_dict['excess_noise'] = parameter_dict['excessNoise']
    argument_dict['merit_function'] = -1#parameter_dict['meritFunction']

    axp.plot(argument_dict=argument_dict)
    axp.argument_dict = argument_dict
    
    return axp
