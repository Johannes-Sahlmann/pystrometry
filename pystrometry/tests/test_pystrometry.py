import copy
from astropy import constants as const
from ..pystrometry import pjGet_a_m_barycentre, pjGet_m2 # keplerian_secondary_mass,
from ..pystrometry import convert_from_linear_to_angular, convert_from_angular_to_linear
import astropy.units as u

import numpy as np

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

