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


def make_comparison_figures(table, parameter_mapping, mapping_dr3id_to_starname,
                            highlight_index=None, description_str='',
                            save_plot=True, plot_dir=os.getcwd()):
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
    discrepancy_table['Name_2'] = table['Name_dedreq-695']
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
        pl.show()
        pl.text(0.01, 0.99, description_str, horizontalalignment='left', verticalalignment='top',
                transform=pl.gca().transAxes)
        if save_plot:
            figure_name = os.path.join(plot_dir, '{}_comparison_to_{}.pdf'.format(miks_field, table.meta['comparison_to']))
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

    pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
    pl.plot(table['p1_a1'], np.abs(table['p1_period'] - table[parameter_mapping['period']]), 'bo')
    pl.xlabel('Fitted semimajor axis (mas)')
    pl.ylabel('Period error (day)')
    pl.show()
    if save_plot:
        figure_name = os.path.join(plot_dir, 'period_error_vs_a1.pdf')
        pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

    return discrepancy_table


def make_orbit_figure(selected_systems, index, epoch_data_dir, mapping_dr3id_to_starname, plot_dir,
                      m1_MS=1.):

    source_id = selected_systems['SourceId'][index]
    t_ref_jd = selected_systems['T0'][index]
    t_ref_mjd = Time(t_ref_jd, format='jd').mjd

    iad = gaia_astrometry.GaiaIad(source_id, epoch_data_dir)
    iad.load_data()


    # iad.epoch_data.write(os.path.join(epoch_data_dir, '{}_OBSERVATION_DATA_SORTED.rdb'.format(source_id)))
    # 1/0
    # iad_mjd = Time(iad.epoch_data[iad.time_column]*365.25, format='jd').mjd
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
        'plx_mas': selected_systems['plx'][index],
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

    print(attribute_dict)
    print(pystrometry.geometric_elements(np.array([selected_systems['p1_{}'.format(key)][index] for key in 'A B F G'.split()])))
    print(pystrometry.mean_anomaly(iad.t_ref_mjd, attribute_dict['Tp_day'], attribute_dict['P_day']))
    # 1/0

    orbit = pystrometry.OrbitSystem(attribute_dict=attribute_dict)
    print(orbit)

    # 1/0
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

    axp = pystrometry.AstrometricOrbitPlotter(plot_dict)
    axp.print_residual_statistics()

    n_curve = 1500
    timestamps_curve_2d = np.linspace(np.min(iad.epoch_data['MJD']), np.max(iad.epoch_data['MJD']), n_curve)
    axp.t_curve_MJD = timestamps_curve_2d
    axp.title = 'Gaia DR3 {} ({})'.format(source_id, mapping_dr3id_to_starname[source_id])
    argument_dict = {'plot_dir': plot_dir, 'ppm_panel': True, 'frame_residual_panel': True,
             'orbit_only_panel': True, 'ppm_description': 'default', 'epoch_omc_description': 'default',
             'orbit_description': 'default', 'arrow_offset_x': +100, 'arrow_offset_y': +100,
             'name_seed': 'DR3_{}_{}'.format(source_id, mapping_dr3id_to_starname[source_id]), 'scan_angle_definition': scan_angle_definition,
                     }

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

    axp.plot(argument_dict=argument_dict)


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

