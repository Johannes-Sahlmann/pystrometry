import copy
from astropy import constants as const
from ..pystrometry import keplerian_secondary_mass, pjGet_a_m_barycentre, pjGet_m2
import astropy.units as u

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