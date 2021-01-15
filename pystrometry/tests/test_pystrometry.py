from collections import OrderedDict
import copy

from astropy import constants as const
from astropy.time import Time
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose

from ..pystrometry import semimajor_axis_barycentre_linear, pjGet_m2, get_ephemeris, semimajor_axis_relative_linear, semimajor_axis_relative_angular
from ..pystrometry import convert_from_linear_to_angular, convert_from_angular_to_linear
from ..pystrometry import thiele_innes_constants, geometric_elements, OrbitSystem, get_cpsi_spsi_for_2Dastrometry


def test_angular_to_linear():
    """Test conversion between milliarcsecond and meter."""

    a_mas = 3.
    absolute_parallax_mas = 100.

    a_recovered_mas = convert_from_linear_to_angular(convert_from_angular_to_linear(a_mas, absolute_parallax_mas), absolute_parallax_mas)
    # print(np.abs(a_mas - a_recovered_mas))
    assert np.abs(a_mas - a_recovered_mas) < 1e-14


def test_get_ephemeris():
    """Test retrieval of ephemeris information from JPL Horizons."""

    start_time = Time(2018., format='jyear')
    stop_time = Time(2019., format='jyear')

    xyzdata = get_ephemeris(start_time=start_time, stop_time=stop_time, step_size='5d',
                  verbose=False, overwrite=True)

    assert len(xyzdata) == 74


def test_keplerian_equations(verbose=False):
    """Test semimajor axis and mass consistency when applying Kepler's equations."""

    MS_kg = const.M_sun.value
    MJ_kg = const.M_jup.value  # jupiter mass in kg

    m1_kg = copy.deepcopy(MS_kg)
    m2_kg = copy.deepcopy(MJ_kg)
    P_day = 400.

    a_m = semimajor_axis_barycentre_linear(m1_kg / MS_kg, m2_kg / MJ_kg, P_day)

    # m2_kg_recovered = keplerian_secondary_mass(m1_kg, a_m, P_day)
    m2_kg_recovered = pjGet_m2(m1_kg, a_m, P_day)

    if verbose:
        print('')
        print(m2_kg/MJ_kg)
        print(m2_kg_recovered/MJ_kg)
        print(m2_kg/m2_kg_recovered - 1 )

    assert m2_kg/m2_kg_recovered - 1 < 1e-14


def test_thiele_innes():
    """Test conversion between Thiele Innes constants and geometric elements."""

    n_grid = 20
    a_mas = np.linspace(0.1, 100, n_grid)
    omega_deg = np.linspace(0.1, 90, n_grid, endpoint=False)
    OMEGA_deg = np.linspace(0.1, 90, n_grid, endpoint=False)
    i_deg = np.linspace(0.5, 180, n_grid, endpoint=False)

    a_mesh, omega_mesh, OMEGA_mesh, i_mesh = np.meshgrid(a_mas, omega_deg, OMEGA_deg, i_deg)
    a = a_mesh.flatten()
    omega = omega_mesh.flatten()
    OMEGA = OMEGA_mesh.flatten()
    i = i_mesh.flatten()

    # input_array = np.array([a_mas, omega_deg, OMEGA_deg, i_deg])
    input_array = np.array([a, omega, OMEGA, i])

    thiele_innes_parameters = thiele_innes_constants(input_array)
    geometric_parameters = geometric_elements(thiele_innes_parameters)

    absolute_tolerance = 1e-6
    assert_allclose(a, geometric_parameters[0], atol=absolute_tolerance)
    assert_allclose(omega, geometric_parameters[1], atol=absolute_tolerance)
    assert_allclose(OMEGA, geometric_parameters[2], atol=absolute_tolerance)
    assert_allclose(i, geometric_parameters[3], atol=absolute_tolerance)


def test_orbit_computation(verbose=False):
    """Perform basic checks on single Keplerian systems.

    The first check is to verify that two orbits that defer by +180deg in omega_deg
    and OMEGA_deg are equivalent in terms of astrometry.

    The effect of that change on the radial velocity alone is the same as replacing i
    by 180deg - i (but the astrometry changes in that case).

    """

    # example orbit system
    attribute_dict = OrderedDict([('RA_deg', 164.9642810928918), ('DE_deg', -21.2190511382063),
                 ('offset_alphastar_mas', 154.35507953), ('offset_delta_mas', -206.09777548),
                 ('absolute_plx_mas', 27.0817358), ('muRA_mas', 105.77971602),
                 ('muDE_mas', -162.05280475), ('rho_mas', -27.13228264),
                 ('Tp_day', 57678.474567094345), ('omega_deg', -23.772188428200618),
                 ('P_day', 687.96689982), ('ecc', 0.0807917), ('OMEGA_deg', 114.45626491875018),
                 ('i_deg', 31.976978902085566), ('delta_mag', 5.74), ('m1_MS', 0.08),
                 ('m2_MJ', 69.38476024572319), ('Tref_MJD', 57622.37084552435),
                 ('scan_angle_definition', 'hipparcos'),
                 ('parallax_correction_mas', 0.8182385292016127)])


    n_grid = 5
    omega_deg = np.linspace(0.1, 180, n_grid, endpoint=False)
    OMEGA_deg = np.linspace(0.1, 180, n_grid, endpoint=False)
    i_deg = np.linspace(0.5, 180, n_grid, endpoint=False)

    omega_mesh, OMEGA_mesh, i_mesh = np.meshgrid(omega_deg, OMEGA_deg, i_deg)
    omega_array = omega_mesh.flatten()
    OMEGA_array = OMEGA_mesh.flatten()
    i_array = i_mesh.flatten()

    for k in range(len(omega_array)):
        attribute_dict['omega_deg'] = omega_array[k]
        attribute_dict['OMEGA_deg'] = OMEGA_array[k]
        attribute_dict['i_deg'] = i_array[k]

        systems = OrderedDict()

        # first orbit
        orbit = OrbitSystem(attribute_dict=attribute_dict)
        if verbose:
            print(orbit)

        # seconf modified orbit
        attribute_dict2 = copy.deepcopy(attribute_dict)
        attribute_dict3 = copy.deepcopy(attribute_dict)
        attribute_dict2['OMEGA_deg'] += 180.
        attribute_dict2['omega_deg'] += 180.
        orbit2 = OrbitSystem(attribute_dict=attribute_dict2)
        if verbose:
            print(orbit2)

        attribute_dict3['i_deg'] = 180. - attribute_dict3['i_deg']
        orbit3 = OrbitSystem(attribute_dict=attribute_dict3)
        if verbose:
            print(orbit3)

        systems[0] = {'orbit_system': orbit}
        systems[1] = {'orbit_system': orbit2}
        systems[2] = {'orbit_system': orbit3}

        n_orbit = 2
        n_curve = 100

        # compute timeseries for both systems
        for i, system in systems.items():
            orbit_system = system['orbit_system']

            timestamps_curve_2D = np.linspace(orbit_system.Tp_day - orbit_system.P_day,
                                              orbit_system.Tp_day + n_orbit + orbit_system.P_day,
                                              n_curve)


            timestamps_curve_1D, cpsi_curve, spsi_curve, xi_curve, yi_curve = get_cpsi_spsi_for_2Dastrometry(timestamps_curve_2D)

            # relative orbit
            phi0_curve_relative = orbit_system.relative_orbit_fast(timestamps_curve_1D, spsi_curve, cpsi_curve,
                                                           shift_omega_by_pi=True)

            systems[i]['relative_orbit'] = phi0_curve_relative
            systems[i]['rv_orbit'] = orbit_system.compute_radial_velocity(timestamps_curve_2D)
            systems[i]['barycentric_orbit'] =  orbit_system.pjGetBarycentricAstrometricOrbitFast(timestamps_curve_1D,
                                                                          spsi_curve, cpsi_curve)


        absolute_tolerance = 1e-6

        # check that relative orbits are the same
        assert_allclose(systems[0]['relative_orbit'], systems[1]['relative_orbit'], atol=absolute_tolerance)

        # check that barycentric orbits are the same
        assert_allclose(systems[0]['barycentric_orbit'], systems[1]['barycentric_orbit'], atol=absolute_tolerance)

        # check that RV orbits are the same except for the sign
        if (systems[0]['orbit_system'].gamma_ms ==0) and (systems[1]['orbit_system'].gamma_ms ==0):
            assert_allclose(systems[0]['rv_orbit'], -1*systems[1]['rv_orbit'], atol=absolute_tolerance)
            assert_allclose(systems[0]['rv_orbit'], systems[2]['rv_orbit'], atol=absolute_tolerance)


def test_default_orbit(verbose=False):
    """Perform basic checks on single Keplerian systems."""

    orb = OrbitSystem()
    times_mjd = np.array([40672.5])
    orb.ppm(times_mjd)

    # test photocentre orbit
    timestamps_curve_1d, cpsi_curve, spsi_curve, xi_curve, yi_curve = get_cpsi_spsi_for_2Dastrometry(times_mjd)
    assert_allclose(orb.photocenter_orbit(timestamps_curve_1d, cpsi_curve, spsi_curve),
                    [-1.57307485e-03, -6.08159828e-19], rtol=1e-9)


def test_semimajor_axes():
    m1_mjup = (const.M_earth / const.M_jup).value
    p_day = u.year.to(u.day)
    d_pc = 10.
    a_relative_m = semimajor_axis_relative_linear(1.0, m1_mjup, p_day)
    assert_allclose(a_relative_m*u.m.to(u.AU), 1, atol=1e-4)

    assert_allclose(semimajor_axis_relative_angular(1.0, m1_mjup, p_day, d_pc), 10, atol=1e-3)