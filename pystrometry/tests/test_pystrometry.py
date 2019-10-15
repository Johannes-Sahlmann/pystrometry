import copy

from astropy import constants as const
import numpy as np
from numpy.testing import assert_allclose


from ..pystrometry import pjGet_a_m_barycentre, pjGet_m2 # keplerian_secondary_mass,
from ..pystrometry import convert_from_linear_to_angular, convert_from_angular_to_linear
from ..pystrometry import thiele_innes_constants, geometric_elements


def test_angular_to_linear():
    a_mas = 3.
    absolute_parallax_mas = 100.

    a_recovered_mas = convert_from_linear_to_angular(convert_from_angular_to_linear(a_mas, absolute_parallax_mas), absolute_parallax_mas)
    print(np.abs(a_mas - a_recovered_mas))
    assert np.abs(a_mas - a_recovered_mas) < 1e-14


def test_keplerian_equations():
    MS_kg = const.M_sun.value
    MJ_kg = const.M_jup.value  # jupiter mass in kg

    m1_kg = copy.deepcopy(MS_kg)
    m2_kg = copy.deepcopy(MJ_kg)
    P_day = 400.

    a_m = pjGet_a_m_barycentre(m1_kg/MS_kg, m2_kg/MJ_kg, P_day)
    print('')

    # m2_kg_recovered = keplerian_secondary_mass(m1_kg, a_m, P_day)
    m2_kg_recovered = pjGet_m2(m1_kg, a_m, P_day)

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
