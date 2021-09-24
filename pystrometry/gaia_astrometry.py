"""Module to support working with Gaia epoch astrometry data.

Authors
-------

    Johannes Sahlmann



"""

from collections import OrderedDict
import os

from astropy.table import Table
from astropy.time import Time
import numpy as np

from .pystrometry import OrbitSystem, convert_from_angular_to_linear, pjGet_m2, AstrometricOrbitPlotter, MS_kg, MJ_kg

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

        # sort by time
        self.epoch_data.sort(self._time_field)

        # identify focal plane transits
        unique_transit_ids = np.unique(self.epoch_data[self._transit_id_field])
        self.n_fov_transit = len(unique_transit_ids)
        self.epoch_data[self._fov_transit_id_field] = np.ones(len(self.epoch_data)).astype(int)
        for ob_number, transit_id in enumerate(unique_transit_ids):
            t_index = np.where(self.epoch_data[self._transit_id_field] == transit_id)[0]
            self.epoch_data[self._fov_transit_id_field][t_index] = ob_number + 1

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
        print('RSE cut removed {} frames (of {})'.format(self.n_filtered_frames,
                                                         self.n_original_frames))
        self.epoch_data.remove_rows(remove_index)


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

        if ('sigma_p1_a1_mas' in parameter_dict.keys()) and (parameter_dict['Nplanets'] == 1):
            # temporary: only for single-companion solutions
            orbit_descr = '$\\alpha={0[p1_a1_mas]:2.{prec}f}\\pm{0[sigma_p1_a1_mas]:2.{prec}f}$ mas (ratio={0[p1_a1_div_sigma_a1_mas]:2.1f})\n'.format(
                dict(parameter_dict), prec=3)
            orbit_descr += '$P={0[p1_period_day]:2.{prec}f}\\pm{0[sigma_p1_period_day]:2.{prec}f}$ d\n'.format(
                dict(parameter_dict), prec=1)
            orbit_descr += '$e={0[p1_ecc]:2.{prec}f}\\pm{0[sigma_p1_ecc]:2.{prec}f}$\n'.format(
                dict(parameter_dict), prec=3)
            orbit_descr += '$i={0[p1_incl_deg]:2.{prec}f}$ deg\n'.format(dict(parameter_dict),
                                                                         prec=2)
            orbit_descr += '$\\omega={0[p1_omega_deg]:2.{prec}f}$ deg\n'.format(
                dict(parameter_dict), prec=2)
            orbit_descr += '$\\Omega={0[p1_OMEGA_deg]:2.{prec}f}$ deg\n'.format(
                dict(parameter_dict), prec=2)
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
                     'ppm_description'      : 'default', 'epoch_omc_description': 'default',
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

