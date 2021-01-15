"""Utility functions to support DU437 activities.

Authors
-------

    Johannes Sahlmann

"""

import copy
import os

from astropy.table import Table
from astropy.time import Time
import numpy as np
import pylab as pl

from .. import gaia_astrometry, pystrometry

try:
    from helpers.table_helpers import plot_columns_simple
except ImportError:
    print('universal_helpers not available')


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
    discrepancy_table['SourceId'] = table['SourceId']
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
        pl.ylabel('{} (MIKS-GA)'.format(miks_field))
        pl.title('{} sources from {}'.format(len(table), table.meta['comparison_to']))
        pl.text(0.01, 0.99, description_str, horizontalalignment='left', verticalalignment='top',
                transform=pl.gca().transAxes)
        pl.show()
        if save_plot:
            figure_name = os.path.join(plot_dir, '{}_comparison_to_{}.pdf'.format(miks_field, table.meta['comparison_to']))
            pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)


    # period phase error:


    miks_name = 'period'
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


    alpha_mas = selected_systems['p1_a1'][index]
    absolute_parallax_mas = selected_systems['plx'][index]
    a_m = pystrometry.convert_from_angular_to_linear(alpha_mas, absolute_parallax_mas)
    P_day = selected_systems['p1_period'][index]
    m2_kg = pystrometry.pjGet_m2(m1_MS*pystrometry.MS_kg, a_m, P_day)
    m2_MJ = m2_kg/pystrometry.MJ_kg

    attribute_dict = {
        'offset_alphastar_mas': selected_systems['alphaStarOffset'][index],
        'offset_delta_mas': selected_systems['deltaOffset'][index],
        # 'RA_deg': 0.,
        # 'DE_deg': 0.,
        'RA_deg': selected_systems['alpha0'][index],
        'DE_deg': selected_systems['delta0'][index],
        # 'plx_mas': selected_systems['plx'][index],
        'absolute_plx_mas': selected_systems['plx'][index],
        'muRA_mas': selected_systems['muAlphaStar'][index],
        'muDE_mas': selected_systems['muDelta'][index],
        'P_day': selected_systems['p1_period'][index],
        'ecc': selected_systems['p1_ecc'][index],
        'omega_deg': selected_systems['p1_omega'][index],
        'OMEGA_deg': selected_systems['p1_OMEGA'][index],
        'i_deg': selected_systems['p1_incl'][index],
        'a_mas': selected_systems['p1_a1'][index],
        'Tp_day': t_ref_mjd + selected_systems['p1_Tp'][index],
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

    source_id = selected_systems['SourceId'][index]
    t_ref_jd = selected_systems['T0'][index]

    iad = get_gaia_iad(source_id, t_ref_jd, epoch_data_dir, verbose=verbose)

    orbit = make_orbit_system(selected_systems, index, iad.scan_angle_definition, iad.t_ref_mjd, m1_MS=m1_MS,
                              degenerate_orbit=degenerate_orbit, verbose=verbose)

    # set coeffMatrix in orbit object
    ppm_signal_mas = orbit.ppm(iad.epoch_data['MJD'], psi_deg=np.rad2deg(
        np.arctan2(iad.epoch_data['spsi_obs'], iad.epoch_data['cpsi_obs'])),
                               offsetRA_mas=selected_systems['alphaStarOffset'][index],
                               offsetDE_mas=selected_systems['deltaOffset'][index],
                               externalParallaxFactors=iad.epoch_data['ppfact_obs'], verbose=True)

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
             'orbit_only_panel': True, 'ppm_description': 'default', 'epoch_omc_description': 'default',
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

    source_id = selected_systems['SourceId'][index]
    t_ref_jd = selected_systems['T0'][index]
    t_ref_mjd = Time(t_ref_jd, format='jd').mjd

    if epoch_data_suffix is None:
        iad = gaia_astrometry.GaiaIad(source_id, epoch_data_dir)
    else:
        iad = gaia_astrometry.GaiaIad(source_id, epoch_data_dir, epoch_data_suffix=epoch_data_suffix)

    iad.load_data()

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

    alpha_mas = selected_systems['p1_a1'][index]
    absolute_parallax_mas = selected_systems['plx'][index]
    a_m = pystrometry.convert_from_angular_to_linear(alpha_mas, absolute_parallax_mas)
    P_day = selected_systems['p1_period'][index]
    m2_kg = pystrometry.pjGet_m2(m1_MS*pystrometry.MS_kg, a_m, P_day)
    m2_MJ = m2_kg/pystrometry.MJ_kg

    attribute_dict = {
        'offset_alphastar_mas': selected_systems['alphaStarOffset'][index],
        'offset_delta_mas': selected_systems['deltaOffset'][index],
        # 'RA_deg': 0.,
        # 'DE_deg': 0.,
        'RA_deg': selected_systems['alpha0'][index],
        'DE_deg': selected_systems['delta0'][index],
        # 'plx_mas': selected_systems['plx'][index],
        'absolute_plx_mas': selected_systems['plx'][index],
        'muRA_mas': selected_systems['muAlphaStar'][index],
        'muDE_mas': selected_systems['muDelta'][index],
        'P_day': selected_systems['p1_period'][index],
        'ecc': selected_systems['p1_ecc'][index],
        'omega_deg': selected_systems['p1_omega'][index],
        'OMEGA_deg': selected_systems['p1_OMEGA'][index],
        'i_deg': selected_systems['p1_incl'][index],
        'a_mas': selected_systems['p1_a1'][index],
        'Tp_day': iad.t_ref_mjd + selected_systems['p1_Tp'][index],
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

    orbit = pystrometry.OrbitSystem(attribute_dict=attribute_dict)
    print(orbit)

    # set coeffMatrix in orbit object
    ppm_signal_mas = orbit.ppm(iad.epoch_data['MJD'], psi_deg=np.rad2deg(
        np.arctan2(iad.epoch_data['spsi_obs'], iad.epoch_data['cpsi_obs'])),
                               offsetRA_mas=selected_systems['alphaStarOffset'][index], offsetDE_mas=selected_systems['deltaOffset'][index],
                               externalParallaxFactors=iad.epoch_data['ppfact_obs'], verbose=True)


    plot_dict = {}
    plot_dict['model_parameters'] = {0: attribute_dict}
    plot_dict['linear_coefficients'] = {'matrix': orbit.coeffMatrix} # dict ('matrix', 'table')
    plot_dict['data_type'] = '1d'
    if hasattr(iad, 'xi'):
        plot_dict['data_type'] = 'gaia_2d'
    else:
        plot_dict['data_type'] = '1d'
    plot_dict['scan_angle_definition'] = scan_angle_definition
    # plot_dict['xi'] = iad.epoch_data['xi']  # AL indices
    # plot_dict['yi'] = iad.epoch_data['yi']  # AC indices



    for key in iad.epoch_data.colnames:
        if '_obs' in key:
            new_key = key.replace('_obs', '')
            if new_key == 'errda_mas':
                new_key = 'sigma_da_mas'
            iad.epoch_data[new_key] = iad.epoch_data[key]

    plot_dict['data'] = iad

    iad.epoch_data.pprint()

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
        name_seed = 'DR3_{}_{}'.format(source_id, mapping_dr3id_to_starname[source_id])
    else:
        name_seed = 'DR3_{}'.format(source_id)

    argument_dict = {'plot_dir': plot_dir, 'ppm_panel': True, 'frame_residual_panel': True,
             'orbit_only_panel': True, 'ppm_description': 'default', 'epoch_omc_description': 'default',
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
        if colname != 'SourceId':
            if np.any(np.abs(data[colname]) > threshold):
                data.remove_rows(np.where(np.abs(data[colname]) > threshold)[0])

    print('Eliminated {} of {} rows with values > {:1.0e}'.format(len(units_table)-len(data), len(data), threshold))



    plot_columns_simple(data, os.path.join(out_dir, 'best_solution_simple_plots'),
                        name_seed='best_solution', units=units)

