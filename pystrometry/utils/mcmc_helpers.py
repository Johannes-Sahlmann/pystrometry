"""Module to support MCMC analyses of astrometric orbits.

Authors
-------
    - Johannes Sahlmann

"""

import numpy as np

def encode_eccentricity_omega(eccentricity, omega_deg):
    """Convert to alternative parameter set ot mitigate correlations and allow ecc=0.

    Parameters
    ----------
    eccentricity : float
    omega_deg : float
        Angle in degrees

    Returns
    -------
    x1, x2 : tuple of floats
        sqrt(ecc) * sin(omega), sqrt(ecc) * cos(omega)

    """
    omega_rad = np.deg2rad(omega_deg)
    x1 = np.sqrt(eccentricity) * np.sin(omega_rad)
    x2 = np.sqrt(eccentricity) * np.cos(omega_rad)

    return x1, x2


def decode_eccentricity_omega(x1, x2):
    """Convert back to original parameter set.

    Parameters
    ----------
    x1 : float
        sqrt(ecc) * sin(omega)
    x2: float
        sqrt(ecc) * cos(omega)

    Returns
    -------
    eccentricity : float
        Orbital eccentricity
    omega_deg : float
        omega in degrees in the range(-180, 180]

    """
    eccentricity = x1**2 + x2**2
    omega_deg = np.rad2deg(np.arctan2(x1, x2))

    return eccentricity, omega_deg