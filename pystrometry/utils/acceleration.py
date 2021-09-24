import numpy as np
import astropy.units as u

# constants according to BAS-045 Eq. 1
T_A_day = 1035. / 2 / np.sqrt(3)
T_B_day = 1035. / 2

year2day = u.year.to(u.day)

def offset_7p(accel, tau):
    """Offset function corresponding to acceleration term, see BAS-045 Eq. 2

    Note that accel is given in mas/year**2 whereas tau is in days. Therefore a conversion factor
    has to be applied at the end.

    Parameters
    ----------
    accel : float
        accel_ra or accel_dec in [mas/year**2]
    tau : float
        reference_time corrected time in days (\tilde tau = t - \tilde T)

    Returns
    -------

    """
    return 1. / 2 * accel * (tau ** 2 - T_A_day ** 2) / year2day**2


def offset_9p(deriv_accel, tau):
    """Offset function corresponding to acceleration term, see BAS-045 Eq. 2

    Parameters
    ----------
    deriv_accel : float
        deriv_accel_ra or deriv_accel_dec in [mas/year**3]
    tau : float
        reference_time corrected time in days (\tilde tau = t - \tilde T)

    Returns
    -------

    """
    return 1. / 6 * deriv_accel * (tau ** 2 - T_B_day ** 2) * tau / year2day**3
