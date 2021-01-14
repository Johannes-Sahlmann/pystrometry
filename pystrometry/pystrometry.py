"""
Classes and functions for high-precision astrometry timeseries analysis.

Authors
-------

    - Johannes Sahlmann


Notes
-----
    - should support python 2.7 and 3.5 (for the time being)


"""


from __future__ import print_function

import copy
import os
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
from astropy import constants as const
from astropy.table import Table, Column
import astropy.units as u
from scipy.interpolate import *
import pdb
from astropy.time import Time, TimeDelta
from astropy.table import vstack as tablevstack
from astropy.table import hstack as tablehstack
from astroquery.simbad import Simbad
import sys
if sys.version_info[0] == 3:
    # import urllib.request as urllib
    from urllib.request import urlopen
    from urllib.error import HTTPError
import pickle

import sympy as sp
from scipy.optimize import fmin as scipyfmin

from linearfit import linearfit

try:
    import pyslalib as sla
except (ImportError):
    pass

from .utils import mcmc_helpers

#***********************************CONSTANTS***********************************
global MS_kg, MJ_kg
MS_kg = const.M_sun.value
# MJ_kg = const.M_jup.value
Ggrav = const.G.value
day2sec = u.day.to(u.second)
AU_m = const.au.value
pc_m = const.pc.value # parsec in meters
MJ_kg = const.M_jup.value # jupiter mass in kg
ME_kg = const.M_earth.value
deg2rad = u.deg.to(u.rad)
rad2mas = u.rad.to(u.arcsec)*1000.
deg2as = u.deg.to(u.arcsec)
year2day = u.year.to(u.day)
MJ2MS = MJ_kg/MS_kg

DEFAULT_EPHEMERIS_DICTIONARY = {'Spitzer': 'horizons_XYZ_2003-2020_EQUATORIAL_Spitzer_1day_csv',
'HST'                                    : 'horizons_XYZ_1990-2016_EQUATORIAL_HST_1day_csv',
'WISE'                                   : 'horizons_XYZ_2009-2016_EQUATORIAL_WISE_1day_csv',
'JWST'                                   : 'horizons_XYZ_2012-2023_EQUATORIAL_JWST_1day_csv',
'L2'                                     : 'horizons_XYZ_1990-2035_EQUATORIAL_L2_1day_csv',
'Earth'                                  : 'horizons_XYZ_1990-2035_EQUATORIAL_Eart1day_csv'}

local_dir = os.path.dirname(os.path.abspath(__file__))

global ephemeris_dir
try:
    ephemeris_dir = os.environ['EPHEMERIS_DIRECTORY']
except KeyError:
    ephemeris_dir = os.path.join(local_dir, 'data')


def fractional_luminosity(mag1, mag2):
    """
    defining fraction luminosity of masses M1 and M2 as beta = L2/(L1+L2) and
    mag1-mag2=-2.5 log10(L1/L2), we find
    beta = 1/(1+10^(mag2-mag1))

    :param mag1:
    :param mag2:
    :return:
    """
    return 1./(1. + 10.**(0.4*(mag2-mag1)))

def luminosity_ratio(fractional_lum):
    """Return luminosity ratio S=L2/L1."""

    return fractional_lum / (1 - fractional_lum)


def fractional_mass(m1, m2):
    """
    computes fractional mass
    getB(m1,m2) returns m2/(m1+m2)

    :param m1:
    :param m2:
    :return:
    """
    return m2/(m1+m2)


def periastron_time(lambda_ref_deg, omega_deg, t_ref_mjd, p_day):
    """Return time of periastron passage.


    Parameters
    ----------
    lambda_ref_deg : float
        mean_longitude_at_reference_time
    omega_deg : float
        argument of periastron
    t_ref_mjd : float
        reference time in MJD (e.g. mid-time of observations)
    p_day : float
        orbital period

    Returns
    -------

    """
    # mean anomaly at reference date
    m_ref_deg = lambda_ref_deg - omega_deg

    # phase at pericentre passage
    # phi0_1 = - np.deg2rad(m_ref_deg)/2./np.pi

    # Tp_day = phi0_1 * P_day + TRef_MJD
    # time at periastron
    t_periastron_mjd = t_ref_mjd - p_day * np.deg2rad(m_ref_deg) / (2*np.pi)

    return t_periastron_mjd


def mean_longitude(t_periastron_mjd, omega_deg, t_mjd, p_day):
    """Return mean longitude at time t_mjd.

    Parameters
    ----------
    t_periastron_mjd : float
        time of periastron passage in MJD
    omega_deg : float
        argument of periastron
    t_ref_mjd : float
        reference time in MJD (e.g. mid-time of observations)
    p_day : float
        orbital period

    Returns
    -------
    lambda_deg

    """

    # mean anomaly
    # m_deg = np.rad2deg((t_mjd - t_periastron_mjd) * (2 * np.pi)/p_day)
    m_deg = mean_anomaly(t_mjd, t_periastron_mjd, p_day)

    # mean longitude
    lambda_deg = m_deg + omega_deg

    return lambda_deg


class OrbitSystem(object):
    """Representation of a binary system following Keplerian motion.

    The primary (m1) is typically the brighter component, i.e.
    delta_mag = mag2-mag1 is positive. For cases, where the
    secondary is more massive, the mass ratio q=m2/m1 > 1.


    Notes
    -----
        These features are supported:
        - Differential chromatic refraction
        - Hipparcos and Gaia scan angle definitions


    References
    ----------
        - Started by JSA 2014-01-29
        - Streamlined init by OJO


    """
    def __init__(self, attribute_dict={}):
        """The default attribute values are stored in the hardcoded
        dictionary below, which also defines the list of acceptable
        attributes.

        The content of attribute_dict is transferred to the instance.

        Parameters
        ----------
        attribute_dict : dict
        """
        self.attribute_dict = attribute_dict
        default_dict = {'P_day': 100, 'ecc': 0, 'm1_MS': 1, 'm2_MJ': 1,
                        'omega_deg': 0., 'OMEGA_deg': 0., 'i_deg': 90.,
                        'Tp_day': 0., 'RA_deg': 0., 'DE_deg': 0.,
                        'absolute_plx_mas': 25.,
                        'parallax_correction_mas': 0.,
                        'muRA_mas': 20., 'muDE_mas': 50.,
                        'gamma_ms': 0., 'rvLinearDrift_mspyr': None,
                        'rvQuadraticDrift_mspyr': None,
                        'rvCubicDrift_mspyr': None, 'Tref_MJD': None,
                        'scan_angle_definition': 'hipparcos',
                        'rho_mas': None,  # DCR coefficient
                        'd_mas': None,  # DCR coefficient (if DCR corrector is used)
                        'a_mas': None,
                        'offset_alphastar_mas': 0.,
                        'offset_delta_mas': 0.,
                        'alpha_mas': None,  # photocenter semimajor axis,
                        'delta_mag': None,  # magnitude difference between components
                        'nuisance_x': None,  # nuisance parameters used when performing MCMC analyses 
                        'nuisance_y': None,  # nuisance parameters used when performing MCMC analyses
                        'esinw': None,  # sqrt(ecc) * sin(omega), alternative variable set for MCMC
                        'ecosw': None,  # sqrt(ecc) * 'plx_macos(omega)
                        'm2sini': None,  # sqrt(m2_MJ) * sin(inclination), alternative variable set for MCMC
                        'm2cosi': None,  # sqrt(m2_MJ) * cos(inclination)
                        'lambda_ref': None  # mean longitude at reference time, substitute for time of periastron
                        }

        # Assign user values as attributes when present, use defaults if not
        attribute_keys = attribute_dict.keys()
        for key, val in default_dict.items():
            if key in attribute_keys:
                if key == 'm2_MJ':
                    setattr(self, '_' + key, attribute_dict[key])
                else:
                    setattr(self, key, attribute_dict[key])
            else:
                if key == 'm2_MJ':
                    key = '_' + key
                setattr(self, key, val)

        # Warn users if a key in attribute_dict isn't a default attribute
        mismatch = [key for key in attribute_dict.keys()
                    if key not in default_dict.keys()]
        if mismatch:
            raise KeyError('Key{0} {1} {2} absent in default OrbitClass'
                           .format('s' if len(mismatch) > 1 else '',
                                   mismatch,
                                   'are' if len(mismatch) > 1 else 'is'))

        # decode alternative parameter sets
        if ('esinw' in attribute_keys) and (self.esinw is not None):
            self.ecc, self.omega_deg = mcmc_helpers.decode_eccentricity_omega(self.esinw, self.ecosw)
        if ('m2sini' in attribute_keys) and (self.m2sini is not None):
            self.m2_MJ, self.i_deg = mcmc_helpers.decode_eccentricity_omega(self.m2sini, self.m2cosi)
            self._m2_MJ = self.m2_MJ

        if ('lambda_ref' in attribute_keys) and (self.lambda_ref is not None):
            if self.Tref_MJD is None:
                raise AttributeError('When lambda_ref is used, the reference time Tref_MJD needs to be set!')
            self.Tp_day = periastron_time(self.lambda_ref, self.omega_deg, self.Tref_MJD, self.P_day)

        # treatment of diluted systems
        if ('delta_mag' in attribute_keys) and (self.delta_mag is not None) and (self.delta_mag != 0.):
            # set photocenter orbit size
            beta = fractional_luminosity(0., self.delta_mag)
            f = fractional_mass(self.m1_MS, self.m2_MS)
            a_rel_mas = self.a_relative_angular()
            self.alpha_mas = (f - beta) * a_rel_mas
            if self.alpha_mas < 0:
                self.alpha_mas = 0.
        else:
            self.alpha_mas = self.a_barycentre_angular()
            self.a_mas  = self.alpha_mas



    # 0J0: Assign m2_MJ and m2_MS to properties so their values will be linked
    @property
    def m2_MJ(self):
        return self._m2_MJ

    @m2_MJ.setter
    def m2_MJ(self, val):
        self._m2_MJ = val

    @property
    def m2_MS(self):
        return self._m2_MJ * MJ_kg / MS_kg

    @m2_MS.setter
    def m2_MS(self, val):
        self._m2_MJ = val * MS_kg / MJ_kg

    def __repr__(self):
        d_pc = 1. / (self.absolute_plx_mas / 1000.)

        description = '+'*30 + '\n'
        description += 'System parameters:\n'
        description += "Distance is {:2.1f} pc \t Parallax = {:2.1f} mas\n".format(d_pc, self.absolute_plx_mas)

        description += "Primary   mass = {:4.3f} Msol \t = {:4.3f} Mjup\n".format(self.m1_MS, self.m1_MS * MS_kg / MJ_kg)
        description += "Secondary mass = {:4.3f} Msol \t = {:4.3f} Mjup \t = {:4.3f} MEarth\n".format(self.m2_MS, self.m2_MJ, self.m2_MJ * MJ_kg / ME_kg)
        description += "Mass ratio q=m2/m1 = {:4.6f}\n".format(self.m2_MS / self.m1_MS)

        description += 'a1_mas    = {:2.3f}, a_rel_mas = {:2.3f}\n'.format(self.a_barycentre_angular(), self.a_relative_angular())
        if self.delta_mag is not None:
            description += 'alpha_mas = {:2.3f}, delta_mag = {:2.3f}\n'.format(self.alpha_mas, self.delta_mag)
            description += 'fract.lum beta = {:2.4f}, lum.ratio=L2/L1 = {:2.4f}\n'.format(fractional_luminosity(0, self.delta_mag), luminosity_ratio(fractional_luminosity(0, self.delta_mag)))

        description += "Inclination  {:2.1f} deg\n".format(self.i_deg)
        description += "Period is   {:2.1f} day \t Eccentricity = {:2.3f}\n".format(self.P_day, self.ecc)
        description += "omega = {:2.1f} deg, OMEGA = {:2.1f} deg, T_periastron = {:2.1f} day\n".format(self.omega_deg, self.OMEGA_deg, self.Tp_day)
        description += "RV semi-amplitude of primary = {:2.3f} m/s\n".format(self.rv_semiamplitude_mps())

        return description


        
    def pjGetOrbit(self, N, Norbit=None, t_MJD=None, psi_deg=None,
                   verbose=0, returnMeanAnomaly=0, returnTrueAnomaly=0):
        """
        DOCUMENT ARV -- simulate simultaneous 2D-astrometric and RV observations
        written: J. Sahlmann   27.07.2009   ObsGe
        updated: J. Sahlmann   25.01.2016   STScI/ESA

        :param N:
        :param Norbit:
        :param t_MJD:
        :param psi_deg:
        :param verbose:
        :param returnMeanAnomaly:
        :param returnTrueAnomaly:
        :return:
        """

        #**************************SYSTEM*PARAMETERS***************************

        # Get companion mass in units of solar mass
        m2_MS = self.m2_MS
        #m2_MS = self.m2_MJ * MJ_kg/MS_kg # #companion mass in units of SOLAR mass

        #gamma_ms = 0. #systemic velocity / m s^-1
        d_pc  = 1./ (self.absolute_plx_mas/1000.)

        if verbose:
            print("%s " % "++++++++++++++++++++")
            print("Primary   mass = %1.3f Msol \t = %4.3f Mjup "
                  % (self.m1_MS, self.m1_MS*MS_kg/MJ_kg))
            print("Secondary mass = %1.3f Msol \t = %4.3f Mjup \t = %4.3f MEarth " % ( m2_MS, self.m2_MJ, self.m2_MJ*MJ_kg/ME_kg))
            print("Inclination  %1.3f deg " % self.i_deg)
            print("Mass ratio q = %4.6f  " %( m2_MS/self.m1_MS))
            print("Period is   %3.1f day \t Eccentricity = %2.1f " % (self.P_day,self.ecc))
            print("Distance is %3.1f pc \t Parallax = %3.1f mas " % (d_pc, self.absolute_plx_mas))
            print("omega = %2.1f deg, OMEGA = %2.1f deg, T0 = %2.1f day " % (self.omega_deg, self.OMEGA_deg,self.Tp_day))

        omega_rad = np.deg2rad(self.omega_deg)
        OMEGA_rad = np.deg2rad(self.OMEGA_deg)
        i_rad =     np.deg2rad(self.i_deg)

        #*************************SIMULATION*PARAMATERS*************************
        if Norbit is not None:
            t_day = np.linspace(0, self.P_day*Norbit, N) + self.Tref_MJD
        elif t_MJD is not None:
            t_day = t_MJD
            N = len(t_MJD)

        #****************************RADIAL*VELOCITY****************************

        E_rad = eccentric_anomaly(self.ecc, t_day, self.Tp_day, self.P_day) # eccentric anomaly
        M = (Ggrav * (self.m2_MJ * MJ_kg)**3.
             / (self.m1_MS * MS_kg + self.m2_MJ * MJ_kg)**2.) # mass term for the barycentric orbit of the primary mass
        #M = G * ( m1_MS*MS + m2_MJ*MJ ) #relative orbit
        a_m = (M / (4. * np.pi**2.) * (self.P_day * day2sec)**2.)**(1./3.)  # semimajor axis of the primary mass in m
        a_AU = a_m / AU_m #  in AU

        if 0:
            THETA_rad = 2 * np.arctan(np.sqrt((1 + self.ecc) / (1 - self.ecc))
                                      * np.tan(E_rad/2)) #position angle between radius vector and ref
            THETA_rad = np.arctan2(np.cos(THETA_rad), np.sin(THETA_rad))

            k1 = (2. * np.pi * a_m * np.sin(i_rad)
                  / (self.P_day * day2sec * (1. - self.ecc**2)**(1./2.))) #RV semiamplitude
            rv_ms = k1 * (np.cos( THETA_rad + omega_rad ) +
                          self.ecc * np.cos(omega_rad)) + self.gamma_ms #radial velocity in m/s.

        else: # damien's method
            THETA_rad = TrueAnomaly(self.ecc, E_rad)
            k1 = (2. * np.pi * a_m * np.sin(i_rad)
                  / ( self.P_day * day2sec * (1. - self.ecc**2)**(1./2.))) #RV semiamplitude
            a_mps = RadialVelocitiesConstants(k1, omega_rad, self.ecc)
            #print(a_mps)
            rv_ms = (RadialVelocitiesKepler(a_mps[0], a_mps[1],
                                           a_mps[2], THETA_rad)
                     + self.gamma_ms)

        if self.rvLinearDrift_mspyr is not None:
            drift_ms = ((t_day - self.Tref_MJD)
                        / year2day * self.rvLinearDrift_mspyr)
            rv_ms += drift_ms

        if self.rvQuadraticDrift_mspyr is not None:
            drift_ms = (((t_day - self.Tref_MJD) / year2day)**2
                        * self.rvQuadraticDrift_mspyr)
            rv_ms += drift_ms

        if self.rvCubicDrift_mspyr is not None:
            drift_ms = (((t_day - self.Tref_MJD) / year2day)**3
                        * self.rvCubicDrift_mspyr)
            rv_ms += drift_ms

        a_rel_AU = (Ggrav * (self.m1_MS * MS_kg + self.m2_MJ * MJ_kg) / 4.
                    / (np.pi**2.) * (self.P_day * day2sec)**2.)**(1./3.) / AU_m

        if verbose:
            print("Astrometric semimajor axis of Primary: a = %3.3f AU \t %6.3f muas " % (a_AU, a_AU / d_pc * 1.e6))
            print("Relative semimajor axis of Primary: a = %3.3f AU \t %6.2f mas " %(a_rel_AU, a_rel_AU / d_pc * 1.e3))
            print("Radial velocity semi-amplitude: K1 =  %4.2f m/s  " % k1)

        #******************************ASTROMETRY*******************************
        a_rad = np.arctan2(a_m, d_pc * pc_m)
        a_mas = a_rad * rad2mas # semimajor axis in mas

        aRel_mas = np.arctan2(a_rel_AU * AU_m, d_pc * pc_m) * rad2mas # relative semimajor axis in mas
        TIC = thiele_innes_constants([a_mas, self.omega_deg, self.OMEGA_deg, self.i_deg]) #Thiele-Innes constants
        TIC_rel = thiele_innes_constants([aRel_mas, self.omega_deg + 180.,
                                          self.OMEGA_deg, self.i_deg]) #Thiele-Innes constants
        #A = TIC[0] B = TIC[1] F = TIC[2] G = TIC[3]

        if psi_deg is not None:
            # psi_rad = np.deg2rad(psi_deg)
            phi1 = astrom_signal(t_day, psi_deg, self.ecc,
                                 self.P_day, self.Tp_day, TIC)
            phi1_rel = astrom_signal(t_day, psi_deg, self.ecc,
                                     self.P_day, self.Tp_day, TIC_rel)
            phi2 = np.nan
            phi2_rel = np.nan

        else:
            #first baseline  second baseline
            #bspread1 = 0.; bspread2 = 0. #baseline spread around offset in deg
            bstart1 = 0.
            bstart2 = 90.    #baseline offset in deg

            # for FORS aric + CRIRES RV simulation, the aric measurement gives both axis simultaneously
            psi_deg1 = np.ones(N) * bstart1 #array(bstart1,N)
            # psi_rad1 = psi_deg1*deg2rad
            psi_deg2 = np.ones(N) * bstart2
            # psi_rad2 = psi_deg2*deg2rad

            phi1 = astrom_signal(t_day, psi_deg1, self.ecc,
                                 self.P_day, self.Tp_day, TIC)
            phi2 = astrom_signal(t_day, psi_deg2, self.ecc,
                                 self.P_day, self.Tp_day, TIC)
            phi1_rel = astrom_signal(t_day, psi_deg1, self.ecc,
                                     self.P_day, self.Tp_day, TIC_rel)
            phi2_rel = astrom_signal(t_day, psi_deg2, self.ecc,
                                     self.P_day, self.Tp_day, TIC_rel)

        if returnMeanAnomaly:
            m_deg = mean_anomaly(t_day, self.Tp_day, self.P_day)
            M_rad = np.deg2rad(m_deg)
            return [phi1, phi2, t_day, rv_ms, phi1_rel, phi2_rel, M_rad]

        elif returnTrueAnomaly:
            #M_rad = mean_anomaly(t_day,self.Tp_day,self.P_day)
            return [phi1, phi2, t_day, rv_ms, phi1_rel, phi2_rel, THETA_rad, TIC_rel]

        return [phi1, phi2, t_day, rv_ms, phi1_rel, phi2_rel]

    # 0J0: Added a function to calculate apparent proper motion given two times
    def get_inter_epoch_accel(self, t0, t1):
        """
        Get the apparent proper motion of a source from one epoch to another.
        Estimated by using the parameters of the current `OrbitSystem` class
        instance to calculate the difference in proper motions of the source
        from its position at each time, then subtracting one proper motion from
        the other. (Proxy for acceleration.)

        Parameters
        ----------
        t0 : `float`
            The time (in MJD) of the initial astrometric observation.

        t1 : `float`
            The time (in MJD) of the final astrometric observation.

        Returns
        ----------
        accel_a : `float`
            The proper motion difference on the Delta alpha axis of motion.

        accel_d : `float`
            The proper motion difference on the Delta delta axis of motion.

        accel_mag : `float`
            The magnitude of the previous two proper motion differences.
        """
        # The amount of time over which to calculate the derivative of position
        step = TimeDelta(60 * u.second)

        # Make sure user-given times are interpreted in *JD units
        assert (t0 + step).format.endswith('jd', -2), 't0/t1 not in *JD units'

        # Get the values of the user-provided times plus the time step
        t0_plus_step = (t0 + step).value
        t1_plus_step = (t1 + step).value

        # see about editing get_spsi with better indexing instead of xi/yi
        t1D, cpsi, spsi, xi, yi = get_cpsi_spsi_for_2Dastrometry(
            [t0, t0_plus_step, t1, t1_plus_step])

        # Return coordinates of the source at the desired 4 times
        phis = self.pjGetBarycentricAstrometricOrbitFast(t1D, spsi, cpsi)

        # Separate the result into specific ra/dec arrays
        del_alpha = phis[yi]; del_delta = phis[xi]
        #del_alpha = phis[1::2]; del_delta = phis[::2]

        # Calculate change in Delta alpha after the time step at both t0 and t1
        shift_a0 = del_alpha[1] - del_alpha[0]
        shift_a1 = del_alpha[3] - del_alpha[2]

        # Differentiate over time to get proper motions in this coordinate
        # (units of mas/yr)
        pm_a0 = shift_a0 / ((t0_plus_step - t0) / year2day)
        pm_a1 = shift_a1 / ((t1_plus_step - t1) / year2day)

        # Do the same for Delta delta
        shift_d0 = del_delta[1] - del_delta[0]
        shift_d1 = del_delta[3] - del_delta[2]

        pm_d0 = shift_d0 / ((t0_plus_step - t0) / year2day)
        pm_d1 = shift_d1 / ((t1_plus_step - t1) / year2day)

        # Estimate acceleration in each coord by subtracting PM @t0 from PM @t1
        accel_a = pm_a1 - pm_a0
        accel_d = pm_d1 - pm_d0

        # Get the magnitude of acceleration by taking both coords into account
        accel_mag = np.sqrt(accel_a**2 + accel_d**2)

        return accel_a, accel_d, accel_mag


    def a_barycentre_angular(self):
        """Get the semi-major axis, in milliarcseconds, of the primary object's
        orbit around the system barycenter. Relies on parameter values from the
        current OrbitSystem instance.

        Returns
        ----------
        a_barycentre : `float`
            The apparent semi-major axis of the primary, in milliarcseconds.
        """
        return semimajor_axis_barycentre_angular(self.m1_MS, self.m2_MJ, self.P_day, self.absolute_plx_mas)
        # M = (Ggrav * (self.m2_MJ * MJ_kg)**3.
        #      / (self.m1_MS * MS_kg + self.m2_MJ * MJ_kg)**2.) # mass term for the barycentric orbit of the primary mass
        # a_m = (M / (4. * np.pi**2.) * (self.P_day * day2sec)**2.)**(1./3.)  # semimajor axis of the primary mass in m
        # d_pc  = 1. / (self.absolute_plx_mas / 1000.)
        # a_rad = np.arctan2(a_m, d_pc*pc_m)
        # a_mas = a_rad * rad2mas # semimajor axis in mas
        # return a_mas


    def a_barycentre_linear(self):
        """Get the semi-major axis, in meters, of the primary object's orbit
        around the system barycenter. Relies on parameter values from the
        current OrbitSystem instance.

        Returns
        ----------
        a_m_barycentre : `float`
            The physical semi-major axis of the primary, in meters.
        """
        return semimajor_axis_barycentre_linear(self.m1_MS, self.m2_MJ, self.P_day)
        # M = (Ggrav * (self.m2_MJ * MJ_kg)**3.
        #      / (self.m1_MS * MS_kg + self.m2_MJ * MJ_kg)**2.) # mass term for the barycentric orbit of the primary mass
        # a_m = (M / (4. * np.pi**2.) * (self.P_day * day2sec)**2.)**(1./3.)  # semimajor axis of the primary mass in m
        # return a_m


    def a_relative_angular(self):
        """Get the semi-major axis, in milliarcseconds, of the secondary object's
        orbit around the primary. Relies on parameter values from the current
        OrbitSystem instance.

        Returns
        ----------
        a_relative : `float`
            The apparent semi-major axis of the secondary, in milliarcseconds.
        """
        return semimajor_axis_relative_angular(self.m1_MS, self.m2_MJ, self.P_day, self.absolute_plx_mas)
        # a_rel_m = ((Ggrav * (self.m1_MS * MS_kg + self.m2_MJ * MJ_kg)
        #             / 4. / (np.pi**2.)
        #              * (self.P_day * day2sec)**2.)**(1./3.))
        # #M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        # #a_m = ( M / (4. * np.pi**2.) * (P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
        # d_pc  = 1./ (self.absolute_plx_mas / 1000.)
        # a_rel_rad = np.arctan2(a_rel_m, d_pc * pc_m)
        # a_rel_mas = a_rel_rad * rad2mas # semimajor axis in mas
        # return a_rel_mas


    def a_relative_linear(self):
        """Get the semi-major axis, in meters, of the secondary object's orbit
        around the primary. Relies on parameter values from the current
        OrbitSystem instance.

        Returns
        ----------
        a_m_relative : `float`
            The physical semi-major axis of the secondary, in meters.
        """
        return semimajor_axis_relative_linear(self.m1_MS, self.m2_MJ, self.P_day)
        # a_rel_m = ((Ggrav * (self.m1_MS * MS_kg + self.m2_MJ * MJ_kg)
        #             / 4. / (np.pi**2.)
        #             * (self.P_day * day2sec)**2.)**(1./3.))
        # return a_rel_m


    def rv_semiamplitude_mps(self, component='primary'):
        """Return semi-amplitude of radial velocity orbit."""

        if component=='primary':
            M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        elif component == 'secondary':
            M = Ggrav * (self.m1_MS * MS_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the secondary mass

        a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the component mass in m

        k_mps = 2. * np.pi * a_m * np.sin(np.deg2rad(self.i_deg)) / (
        self.P_day * day2sec * (1. - self.ecc ** 2) ** (1. / 2.))  # RV semiamplitude

        return k_mps

    # def pjGetRV(self,t_day):
    def compute_radial_velocity(self, t_day, component='primary'):
        """Compute radial velocity of primary or secondary component in m/s.

         updated: J. Sahlmann   25.01.2016   STScI/ESA
         updated: J. Sahlmann   13.07.2018   STScI/AURA

        Parameters
        ----------
        t_day
        component

        Returns
        -------
        rv_ms : ndarray
            RV in m/s

        """

        # m2_MS = self.m2_MJ * MJ_kg/MS_kg# #companion mass in units of SOLAR mass
        # i_rad =     np.deg2rad(self.i_deg)

        #**************RADIAL*VELOCITY**************************************************
        E_rad = eccentric_anomaly(self.ecc, t_day, self.Tp_day, self.P_day) # eccentric anomaly
        if component=='primary':
            # M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
            omega_rad = np.deg2rad(self.omega_deg)
        elif component == 'secondary':
            # M = Ggrav * (self.m1_MS * MS_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the secondary mass
            omega_rad = np.deg2rad(self.omega_deg + 180.)

        # a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the component mass in m
        # a_AU = a_m / AU_m #  in AU

        # damien's method
        THETA_rad = TrueAnomaly(self.ecc, E_rad)
        # k_m = 2. * np.pi * a_m * np.sin(i_rad) / ( self.P_day*day2sec * (1.-self.ecc**2)**(1./2.) ) #RV semiamplitude
        k_m = self.rv_semiamplitude_mps(component=component)
        a_mps = RadialVelocitiesConstants(k_m, omega_rad, self.ecc)
        rv_ms = RadialVelocitiesKepler(a_mps[0], a_mps[1], a_mps[2], THETA_rad) + self.gamma_ms

        if self.rvLinearDrift_mspyr is not None:
            drift_ms = (t_day - self.Tref_MJD)/year2day * self.rvLinearDrift_mspyr
            rv_ms += drift_ms

        if self.rvQuadraticDrift_mspyr is not None:
            drift_ms = ((t_day - self.Tref_MJD)/year2day)**2 * self.rvQuadraticDrift_mspyr
            rv_ms += drift_ms

        if self.rvCubicDrift_mspyr is not None:
            drift_ms = ((t_day - self.Tref_MJD)/year2day)**3 * self.rvCubicDrift_mspyr
            rv_ms += drift_ms

        return rv_ms

    def get_t_plot(self, time_offset_day=0., n_curve=100, n_orbit=1, format='jyear'):
        """Return an array of times to use for plotting the timeseries

        Parameters
        ----------
        time_offset_day

        Returns
        -------

        """

        t_day = np.linspace(0, self.P_day * n_orbit, n_curve) - self.P_day/2 + self.Tp_day + time_offset_day
        t_plot = getattr(Time(t_day, format='mjd'), format)
        return t_plot


    def plot_rv_orbit(self, component='primary', n_curve=100, n_orbit=1, line_color='k',
                      line_style='-', line_width=1, rv_unit='kmps', time_offset_day=0.,
                      gamma_mps=None, axis=None, plot_parameters_ensemble=None):
        """Plot the radial velocity orbit of the primary

        Returns
        -------

        """

        # if gamma_mps is None:
        #     gamma_mps = self.gamma_ms

        if axis is None:
            axis = pl.gca()

        if rv_unit == 'kmps':
            rv_factor = 1/1000.
        else:
            rv_factor = 1.
        t_day = np.linspace(0, self.P_day * n_orbit, n_curve) - self.P_day/2 + self.Tp_day + time_offset_day
        t_plot = Time(t_day, format='mjd').jyear
        if component=='primary':
            rv_mps = (self.compute_radial_velocity(t_day, component=component)) * rv_factor
            axis.plot(t_plot, rv_mps, ls=line_style, color=line_color, lw=line_width)
            # if plot_parameters_ensemble is not None:
            #     rv_mps = (self.compute_radial_velocity(t_day, component=component)) * rv_factor
            #     1/0
        elif component=='secondary':
            rv_mps = (self.compute_radial_velocity(t_day, component=component)) * rv_factor
            axis.plot(t_plot, rv_mps, ls=line_style, color=line_color, lw=line_width)
        elif component=='both':
            rv_mps_1 = (self.compute_radial_velocity(t_day, component='primary')) * rv_factor
            rv_mps_2 = (self.compute_radial_velocity(t_day, component='secondary')) * rv_factor
            axis.plot(t_plot, rv_mps_1, ls=line_style, color=line_color, lw=line_width+2, label='primary')
            axis.plot(t_plot, rv_mps_2, ls=line_style, color=line_color, lw=line_width, label='secondary')
        elif component=='difference':
            rv_mps_1 = self.compute_radial_velocity(t_day, component='primary') * rv_factor
            rv_mps_2 = self.compute_radial_velocity(t_day, component='secondary') * rv_factor
            axis.plot(t_plot, rv_mps_1-rv_mps_2, ls=line_style, color=line_color, lw=line_width+2, label='difference')



    def pjGetOrbitFast(self, N, Norbit=None, t_MJD=None, psi_deg=None, verbose=0):
    # /* DOCUMENT ARV -- simulate fast 1D astrometry for planet detection limits
    #    written: J. Sahlmann   18 May 2015   ESAC
    # */


        m2_MS = self.m2_MJ * MJ_kg/MS_kg# #companion mass in units of SOLAR mass
        d_pc  = 1./ (self.absolute_plx_mas/1000.)

        omega_rad = np.deg2rad(self.omega_deg)
        OMEGA_rad = np.deg2rad(self.OMEGA_deg)
        i_rad = np.deg2rad(self.i_deg)

        t_day = t_MJD
        N = len(t_MJD)
        #**************ASTROMETRY********************************************************

        M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
        a_AU = a_m / AU_m #  in AU
        a_rel_AU = (Ggrav*(self.m1_MS*MS_kg+self.m2_MJ*MJ_kg) / 4. /(np.pi**2.) *(self.P_day*day2sec)**2.)**(1./3.)/AU_m


        a_rad = np.arctan2(a_m,d_pc*pc_m)
        a_mas = a_rad * rad2mas # semimajor axis in mas
        aRel_mas = np.arctan2(a_rel_AU*AU_m,d_pc*pc_m) * rad2mas # relative semimajor axis in mas

        TIC     = thiele_innes_constants([a_mas   , self.omega_deg     , self.OMEGA_deg, self.i_deg]) #Thiele-Innes constants

        phi1 = astrom_signal(t_day, psi_deg, self.ecc, self.P_day, self.Tp_day, TIC)
        phi1_rel = np.nan #astrom_signal(t_day,psi_deg,self.ecc,self.P_day,self.Tp_day,TIC_rel)
        phi2 = np.nan
        phi2_rel = np.nan
        rv_ms=np.nan

        return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel]


    def pjGetBarycentricAstrometricOrbitFast(self, t_MJD, spsi, cpsi):
        """Simulate fast 1D astrometry for planet detection limits.

        written: J. Sahlmann   18 May 2015   ESAC
        updated: J. Sahlmann   25.01.2016   STScI/ESA
        updated: J. Sahlmann   14.01.2021   RHEA for ESA

        Parameters
        ----------
        t_MJD
        spsi
        cpsi

        Returns
        -------

        """

        # semimajor axis in mas
        a_mas = self.a_barycentre_angular()

        # Thiele-Innes constants
        TIC = thiele_innes_constants([a_mas, self.omega_deg, self.OMEGA_deg, self.i_deg])
        phi1 = astrom_signalFast(t_MJD, spsi, cpsi, self.ecc, self.P_day, self.Tp_day, TIC,
                                 scan_angle_definition=self.scan_angle_definition)
        return phi1


    def photocenter_orbit(self, t_MJD, spsi, cpsi):
        """Return the photocenter displacement at the input times.

        Parameters
        ----------
        t_MJD
        spsi
        cpsi

        Returns
        -------

        """
        if (self.delta_mag is None) or (self.delta_mag == 0):
            return self.pjGetBarycentricAstrometricOrbitFast(t_MJD, spsi, cpsi)
        else:
            relative_orbit_mas = self.relative_orbit_fast(t_MJD, spsi, cpsi, shift_omega_by_pi=False)
            beta = fractional_luminosity(0., self.delta_mag)
            f = fractional_mass(self.m1_MS, self.m2_MS)
            photocentric_orbit_mas = (f - beta) * relative_orbit_mas
            return photocentric_orbit_mas


    def relative_orbit_fast(self, t_MJD, spsi, cpsi, unit='mas', shift_omega_by_pi=True,
                            coordinate_system='cartesian'):
        """
                Simulate fast 1D orbital astrometry
        written: J. Sahlmann   18 May 2015   ESAC
        updated: J. Sahlmann   25.01.2016   STScI/ESA
        updated: J. Sahlmann   27 February 2017   STScI/AURA

        returns relative orbit in linear or angular units

        Parameters
        ----------
        t_MJD
        spsi
        cpsi
        unit
        shift_omega_by_pi
        coordinate_system

        Returns
        -------

        """

        #mass term of relative orbit
        M_rel = Ggrav*(self.m1_MS*MS_kg+self.m2_MJ*MJ_kg)

        # semimajor axis of the relative orbit in m
        a_rel_m = ( M_rel / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)

        # shift argument of periastron relative to barycentric orbit of primary mass M1
        if shift_omega_by_pi:
            omega_rel_deg = self.omega_deg + 180.
        else:
            omega_rel_deg = self.omega_deg

        if unit == 'mas':
            d_pc  = 1./ (self.absolute_plx_mas/1000.)
            a_rad = np.arctan2(a_rel_m,d_pc*pc_m)
            # semimajor axis in mas
            a_rel_mas = a_rad * rad2mas
            a_rel = a_rel_mas
        elif unit == 'meter':
            a_rel = a_rel_m

        #Thiele-Innes constants
        TIC = thiele_innes_constants([a_rel, omega_rel_deg, self.OMEGA_deg, self.i_deg])

        # by default these are cartesian coordinates
        phi1 = astrom_signalFast(t_MJD, spsi, cpsi, self.ecc, self.P_day, self.Tp_day, TIC)

        # convert to polar coordinates if requested
        if coordinate_system=='polar':
            xi = np.where(cpsi==1)[0]
            yi = np.where(cpsi==0)[0]
            rho = np.sqrt(phi1[xi]**2 + phi1[yi]**2)
            phi_deg = np.rad2deg(np.arctan2(phi1[xi], phi1[yi]))%360.
            phi1[xi] = rho
            phi1[yi] = phi_deg

        return phi1

    def ppm(self, t_MJD, psi_deg=None, offsetRA_mas=0, offsetDE_mas=0, externalParallaxFactors=None,
            horizons_file_seed=None, instrument=None, verbose=False):
        """Compute parallax and proper motion.

        Parameters
        ----------
        t_MJD
        psi_deg
        offsetRA_mas
        offsetDE_mas
        externalParallaxFactors
        horizons_file_seed
        instrument
        verbose

        Returns
        -------

        """
        assert isinstance(t_MJD, (list, np.ndarray))

        # check that t_MJD is sorted and increasing
        if sorted(list(t_MJD)) != list(t_MJD):
            raise RuntimeError('Please sort the input timestamps first.')
        if t_MJD[0] > t_MJD[-1]:
            raise RuntimeError('Please sort the input timestamps in increasing order.')

        Nframes = len(t_MJD)
        t_JD = t_MJD + 2400000.5
        if externalParallaxFactors is not None:
            parf = externalParallaxFactors
        else:
            parf = get_parallax_factors(self.RA_deg, self.DE_deg, t_JD, horizons_file_seed=horizons_file_seed,
                                        verbose=verbose, instrument=instrument, overwrite=False)

        self.parf = parf
        if self.Tref_MJD is None:
            self.Tref_MJD = np.mean(t_MJD)

        trel_year = (t_MJD - self.Tref_MJD)/year2day

        # % sin(psi) and cos(psi)
        if psi_deg is not None:
            psi_rad = np.deg2rad(psi_deg)
            spsi = np.sin(psi_rad)
            cpsi = np.cos(psi_rad)
            t = trel_year
        else:
            t, cpsi, spsi, xi, yi = get_cpsi_spsi_for_2Dastrometry(trel_year, scan_angle_definition=self.scan_angle_definition)
        tspsi = t*spsi
        tcpsi = t*cpsi

        if psi_deg is not None:
            if externalParallaxFactors is None:
                ppfact = parf[0] * cpsi + parf[1] * spsi    # see Sahlmann+11 Eq. 1 / 8
            else:
                ppfact = parf
        else:
            ppfact = np.zeros(2*Nframes)
            ppfact[xi] = parf[0]
            ppfact[yi] = parf[1]
            self.xi = np.where(xi)[0]
            self.yi = np.where(yi)[0]

        if self.scan_angle_definition == 'hipparcos':
            C = np.array([cpsi, spsi, ppfact, tcpsi, tspsi])
        elif self.scan_angle_definition == 'gaia':
            C = np.array([spsi, cpsi, ppfact, tspsi, tcpsi])

        self.coeffMatrix = C
        self.timeUsedInTcspsi = np.array(t)
        if psi_deg is not None:
            self.MjdUsedInTcspsi = t_MJD
        else:
            self.MjdUsedInTcspsi = np.array(np.sort(np.tile(t_MJD, 2)))

        parallax_for_ppm_mas = self.absolute_plx_mas - self.parallax_correction_mas

        inVec = np.array([offsetRA_mas, offsetDE_mas, parallax_for_ppm_mas, self.muRA_mas, self.muDE_mas])
        # inVec = np.array([offsetRA_mas, offsetDE_mas, parallax_for_ppm_mas, 0, 0])

        ppm = np.dot(C.T, inVec)
        if psi_deg is not None:
            return ppm
        else:
            ppm2d = [ppm[xi],ppm[yi]]
            return ppm2d


    def plot_orbits(self, timestamps_curve_2D=None, timestamps_probe_2D=None, timestamps_probe_2D_label=None,
                    delta_mag=None, N_orbit=1., N_curve=100, save_plot=False, plot_dir=None,
                    new_figure=True, line_color='k', line_style='-', line_width=1, share_axes=False,
                    show_orientation=False, arrow_offset_x=0, invert_xaxis=True, show_time=True,
                    timeformat='jyear', name_seed='', verbose=False):
        """Plot barycentric, photocentric, and relative orbits in two panels.

        Parameters
        ----------
        timestamps_curve_2D : MJD
        timestamps_probe_2D : MJD
        timestamps_probe_2D_label
        delta_mag
        N_orbit
        N_curve
        save_plot
        plot_dir
        new_figure
        line_color
        line_style
        line_width
        share_axes
        show_orientation
        arrow_offset_x
        invert_xaxis
        show_time
        timeformat
        name_seed
        verbose

        Returns
        -------

        """
        if self.delta_mag is not None:
            delta_mag = self.delta_mag

        if timestamps_curve_2D is None:
            timestamps_curve_2D = np.linspace(self.Tp_day - self.P_day, self.Tp_day + N_orbit + self.P_day, N_curve)

        timestamps_curve_1D, cpsi_curve, spsi_curve, xi_curve, yi_curve = get_cpsi_spsi_for_2Dastrometry(timestamps_curve_2D)
        # relative orbit
        phi0_curve_relative = self.relative_orbit_fast(timestamps_curve_1D, spsi_curve, cpsi_curve, shift_omega_by_pi = True)

        if timestamps_probe_2D is not None:
            timestamps_probe_1D, cpsi_probe, spsi_probe, xi_probe, yi_probe = get_cpsi_spsi_for_2Dastrometry(timestamps_probe_2D)
            phi0_probe_relative = self.relative_orbit_fast(timestamps_probe_1D, spsi_probe, cpsi_probe, shift_omega_by_pi = True)

        if delta_mag is not None:
            # fractional luminosity
            beta = fractional_luminosity( 0. , 0.+delta_mag )
            #     fractional mass
            f = fractional_mass(self.m1_MS, self.m2_MS)

            # photocentre orbit about the system's barycentre
            phi0_curve_photocentre = (f - beta) * self.relative_orbit_fast(timestamps_curve_1D, spsi_curve, cpsi_curve, shift_omega_by_pi = False)
            if timestamps_probe_2D is not None:
                phi0_probe_photocentre = (f - beta) * self.relative_orbit_fast(timestamps_probe_1D, spsi_probe, cpsi_probe, shift_omega_by_pi = False)


        # barycentric orbit of M1
        phi0_curve_barycentre = self.pjGetBarycentricAstrometricOrbitFast(timestamps_curve_1D, spsi_curve, cpsi_curve)
        if timestamps_probe_2D is not None:
            phi0_probe_barycentre = self.pjGetBarycentricAstrometricOrbitFast(timestamps_probe_1D, spsi_probe, cpsi_probe)

        n_figure_columns = 2
        n_figure_rows = 1
        # fig, axes = pl.subplots(n_figure_rows, n_figure_columns, figsize=(n_figure_columns*6, n_figure_rows*5), facecolor='w', edgecolor='k', sharex=True, sharey=True)

        if new_figure:
                fig, axes = pl.subplots(n_figure_rows, n_figure_columns, figsize=(n_figure_columns*6, n_figure_rows*5), facecolor='w', edgecolor='k', sharex=share_axes, sharey=share_axes)
        else:
            axes = pl.gcf().axes
        # plot smooth orbit curve
        axes[0].plot(phi0_curve_barycentre[xi_curve], phi0_curve_barycentre[yi_curve],'k--',lw=line_width, color=line_color, ls=line_style) #, label='Barycentre'
        # plot individual epochs
        if timestamps_probe_2D is not None:
            axes[0].plot(phi0_probe_barycentre[xi_probe], phi0_probe_barycentre[yi_probe],'bo',mfc='0.7', label=timestamps_probe_2D_label)

        if delta_mag is not None:
            axes[0].plot(phi0_curve_photocentre[xi_curve], phi0_curve_photocentre[yi_curve],'k--',lw=1, label='Photocentre')
            if timestamps_probe_2D is not None:
                axes[0].plot(phi0_probe_photocentre[xi_probe],phi0_probe_photocentre[yi_probe],'bo')



        if show_orientation:
            # arrow_index_1 = np.int(N_curve/3.3)
            arrow_index_1 = 3*np.int(N_curve/5)
            arrow_index_2 = arrow_index_1 + 10
            length_factor = 1
            arrow_factor = 2

            # ax = pl.axes()
            arrow_base_x = phi0_curve_barycentre[xi_curve][arrow_index_1]
            arrow_base_y = phi0_curve_barycentre[yi_curve][arrow_index_1]
            arrow_delta_x = phi0_curve_barycentre[xi_curve][arrow_index_2] - arrow_base_x
            arrow_delta_y = phi0_curve_barycentre[yi_curve][arrow_index_2] - arrow_base_y

            axes[0].arrow(arrow_base_x+arrow_offset_x, arrow_base_y, arrow_delta_x*length_factor, arrow_delta_y*length_factor, head_width=0.05*arrow_factor, head_length=0.1*arrow_factor, fc=line_color, ec=line_color) #, head_width=0.05, head_length=0.1

        # plot origin = position of barycentre
        axes[0].plot(0,0,'kx')
        axes[0].axhline(y=0,color='0.7',ls='--',zorder=-50)
        axes[0].axvline(x=0,color='0.7',ls='--',zorder=-50)

        axes[0].set_xlabel('Offset in Right Ascension (mas)')
        axes[0].set_ylabel('Offset in Declination (mas)')
        axes[0].axis('equal')
        if invert_xaxis:
            axes[0].invert_xaxis()
        axes[0].legend(loc='best')
        axes[0].set_title('Bary-/photocentric orbit of M1')

        # second panel
        # plot smooth orbit curve
        axes[1].plot(phi0_curve_relative[xi_curve],phi0_curve_relative[yi_curve],'k-',lw=line_width, color=line_color, ls=line_style)
        # plot individual epochs
        if timestamps_probe_2D is not None:
            axes[1].plot(phi0_probe_relative[xi_probe],phi0_probe_relative[yi_probe], 'bo', label=timestamps_probe_2D_label)
            if verbose:
                print('relative separation: {}'.format(np.linalg.norm([phi0_probe_relative[xi_probe],phi0_probe_relative[yi_probe]], axis=0)))

        if show_orientation:
            # ax = pl.axes()
            arrow_base_x = phi0_curve_relative[xi_curve][arrow_index_1]
            arrow_base_y = phi0_curve_relative[yi_curve][arrow_index_1]
            arrow_delta_x = phi0_curve_relative[xi_curve][arrow_index_2] - arrow_base_x
            arrow_delta_y = phi0_curve_relative[yi_curve][arrow_index_2] - arrow_base_y

            axes[1].arrow(arrow_base_x+arrow_offset_x, arrow_base_y, arrow_delta_x*length_factor, arrow_delta_y*length_factor, head_width=0.05*arrow_factor, head_length=0.1*arrow_factor, fc=line_color, ec=line_color)

        # plot origin = position of primary
        axes[1].plot(0,0,'kx')
        axes[1].axhline(y=0,color='0.7',ls='--',zorder=-50)
        axes[1].axvline(x=0,color='0.7',ls='--',zorder=-50)

        axes[1].set_xlabel('Offset in Right Ascension (mas)')
        axes[1].axis('equal')
        axes[1].legend(loc='best')
        axes[1].set_title('Relative orbit of M2 about M1')
        if (not axes[1]._sharex) and (invert_xaxis):
            axes[1].invert_xaxis()
        pl.show()
        if save_plot:
            fig_name = os.path.join(plot_dir, '{}_orbits_sky.pdf'.format(name_seed))
            plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

        # show barycentric offsets as function of time
        if show_time:
            t_plot_curve = getattr(Time(timestamps_curve_2D, format='mjd'), timeformat)

            n_figure_columns = 2
            n_figure_rows = 1
            fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                    figsize=(n_figure_columns * 8, n_figure_rows * 4),
                                    facecolor='w', edgecolor='k', sharex=share_axes,
                                    sharey=share_axes)
            # plot smooth orbit curve
            axes[0].plot(t_plot_curve, phi0_curve_barycentre[xi_curve], 'k-',
                         lw=line_width, color=line_color, ls=line_style)
            axes[1].plot(t_plot_curve, phi0_curve_barycentre[yi_curve], 'k-',
                         lw=line_width, color=line_color, ls=line_style)

            axes[0].set_ylabel('Offset in Right Ascension (mas)')
            axes[1].set_ylabel('Offset in Declination (mas)')
            axes[0].set_xlabel('Time ({})'.format(timeformat))
            axes[1].set_xlabel('Time ({})'.format(timeformat))

            pl.suptitle('Barycentre orbit')

            # plot individual epochs
            if timestamps_probe_2D is not None:
                axes[0].plot(Time(timestamps_probe_1D[xi_probe], format='mjd').jyear, phi0_probe_barycentre[xi_probe], 'bo',
                             mfc='0.7', label=timestamps_probe_2D_label)
                axes[1].plot(Time(timestamps_probe_1D[yi_probe], format='mjd').jyear, phi0_probe_barycentre[yi_probe], 'bo',
                             mfc='0.7', label=timestamps_probe_2D_label)

            pl.show()
            if save_plot:
                fig_name = os.path.join(plot_dir, '{}_barycentre_orbit_time.pdf'.format(name_seed))
                plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)


    def plot_ppm(self, timestamps_curve_2D=None, timestamps_probe_2D=None,
                    timestamps_probe_2D_label=None,
                    delta_mag=None, N_orbit=1., N_curve=100, save_plot=False, plot_dir=None,
                    new_figure=True, line_color='k', line_style='-', line_width=1, share_axes=False,
                    show_orientation=False, arrow_offset_x=0, invert_xaxis=True, show_time=True,
                    show_difference_to=None, timeformat='jyear',
                    title=None, show_sky=False, name_seed='',
                    **kwargs):
        """Plot the parallax and proper motion of the instance.
        """
        if timestamps_curve_2D is None:
            timestamps_curve_2D = np.linspace(self.Tp_day - self.P_day,
                                              self.Tp_day + N_orbit + self.P_day, N_curve)
        else:
            N_curve = len(timestamps_curve_2D)

        ppm_curve_mas = self.ppm(timestamps_curve_2D, offsetRA_mas=0, offsetDE_mas=0, externalParallaxFactors=None, horizons_file_seed=None, instrument=None, verbose=0)
        if timestamps_probe_2D is not None:
            ppm_probe_mas = self.ppm(timestamps_probe_2D, offsetRA_mas=0, offsetDE_mas=0, externalParallaxFactors=None, horizons_file_seed=None, instrument=None, verbose=0)
        if show_difference_to is not None:
            # expect OrbitSystem instance as input
            ppm_curve_mas_2 = show_difference_to.ppm(timestamps_curve_2D, offsetRA_mas=0, offsetDE_mas=0, externalParallaxFactors=None, horizons_file_seed=None, instrument=None, verbose=0)
            ppm_probe_mas_2 = show_difference_to.ppm(timestamps_probe_2D, offsetRA_mas=0, offsetDE_mas=0, externalParallaxFactors=None, horizons_file_seed=None, instrument=None, verbose=0)

            ppm_curve_mas = [ppm_curve_mas[i] - ppm_curve_mas_2[i] for i in range(len(ppm_curve_mas))]
            ppm_probe_mas = [ppm_probe_mas[i] - ppm_probe_mas_2[i] for i in range(len(ppm_probe_mas))]



        if show_sky:
            n_figure_columns = 1
            n_figure_rows = 1
            if new_figure:
                # fig = pl.figure(figsize=(n_figure_columns * 6, n_figure_rows * 6), facecolor='w', edgecolor='k')
                # axes = pl.gca()
                fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                        figsize=(n_figure_columns * 6, n_figure_rows * 6),
                                        facecolor='w', edgecolor='k', sharex=share_axes,
                                        sharey=share_axes)
                axes = [axes]
            else:
                axes = pl.gcf().axes
            # plot smooth orbit curve
            axes[0].plot(ppm_curve_mas[0], ppm_curve_mas[1], 'k--',
                         lw=line_width, color=line_color, ls=line_style)
            # plot individual epochs
            if timestamps_probe_2D is not None:
                axes[0].plot(ppm_probe_mas[0], ppm_probe_mas[1], 'bo', label=timestamps_probe_2D_label, **kwargs)
            axes[0].set_xlabel('Offset in Right Ascension (mas)')
            axes[0].set_ylabel('Offset in Declination (mas)')
            axes[0].axis('equal')
            if invert_xaxis:
                axes[0].invert_xaxis()
            if show_orientation:
                arrow_index_1 = np.int(N_curve / 5)
                arrow_index_2 = arrow_index_1 + 10
                length_factor = 10
                arrow_factor = 1000

                arrow_base_x = ppm_curve_mas[0][arrow_index_1]
                arrow_base_y = ppm_curve_mas[1][arrow_index_1]
                arrow_delta_x = ppm_curve_mas[0][arrow_index_2] - arrow_base_x
                arrow_delta_y = ppm_curve_mas[2][arrow_index_2] - arrow_base_y

                axes[0].arrow(arrow_base_x + arrow_offset_x, arrow_base_y,
                              arrow_delta_x * length_factor, arrow_delta_y * length_factor,
                              head_width=0.05 * arrow_factor, head_length=0.1 * arrow_factor,
                              fc=line_color,
                              ec=line_color)  # , head_width=0.05, head_length=0.1

            pl.show()
            if save_plot:
                fig_name = os.path.join(plot_dir, '{}_ppm_sky.pdf'.format(name_seed))
                plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

        if show_time:
            n_figure_columns = 2
            n_figure_rows = 1
            if new_figure:
                fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                        figsize=(n_figure_columns * 8, n_figure_rows * 4),
                                        facecolor='w', edgecolor='k', sharex=share_axes,
                                        sharey=share_axes)
            else:
                axes = pl.gcf().axes

            t_plot_curve = getattr(Time(timestamps_curve_2D, format='mjd'), timeformat)

            # plot smooth PPM curve
            axes[0].plot(t_plot_curve, ppm_curve_mas[0], 'k-', lw=line_width, color=line_color, ls=line_style)  # , label='Barycentre'
            axes[1].plot(t_plot_curve, ppm_curve_mas[1], 'k-', lw=line_width, color=line_color, ls=line_style)  # , label='Barycentre'
            axes[0].axhline(y=0, color='0.7', ls='--', zorder=-50)
            axes[1].axhline(y=0, color='0.7', ls='--', zorder=-50)

            axes[0].set_ylabel('Offset in Right Ascension (mas)')
            axes[1].set_ylabel('Offset in Declination (mas)')

            axes[0].set_xlabel('Time ({})'.format(timeformat))
            axes[1].set_xlabel('Time ({})'.format(timeformat))

            if title is not None:
                pl.suptitle(title)

            # plot individual epochs
            if timestamps_probe_2D is not None:
                t_plot_probe = getattr(Time(timestamps_probe_2D, format='mjd'), timeformat)
                axes[0].plot(t_plot_probe, ppm_probe_mas[0], 'bo', label=timestamps_probe_2D_label, **kwargs)
                axes[1].plot(t_plot_probe, ppm_probe_mas[1], 'bo', label=timestamps_probe_2D_label, **kwargs)
                if timestamps_probe_2D_label is not None:
                    axes[0].legend(loc='best')

            pl.show()
            if save_plot:
                fig_name = os.path.join(plot_dir, '{}_ppm_time.pdf'.format(name_seed))
                plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)


class PpmPlotter(object):
    """
    A class to plot results of astrometric fitting of parallax + proper motion

    Attributes
    ----------
    p : array
        holding best fit parameters of linear fit (usually positions,parallax,proper motion)
        part of what linfit returns
    C : matrix
        Numpy Matrix holding the parameters of the linear model
    xmlFileName : string
        filename used to write file on disk

    Methods
    -------
    printSchemaNames()
        prints names of availabe schemas
    getTableNames(schemaName,verbose=0):
        return table names of a certain schema
    """

    def __init__(self, p, C, T, xi, yi, omc, noParallaxFit=0, psi_deg=None, epoch_outlier_dir=None,
                 outlier_sigma_threshold=2., absolute_threshold=None):

        self.p = p
        self.C = C
        self.T = T
        self.xi = xi
        self.yi = yi
        self.omc = omc
        self.noParallaxFit = noParallaxFit
        self.psi_deg = psi_deg

        # compute positions at measurement dates according to best-fit model p (no DCR)
        inVec = p.flatten()[0:5]
        self.ppm_model = np.dot(C[0:len(inVec), :].T, inVec)
        DCR = None

        # compute measured positions (DCR-corrected)
        if C.shape[0] == 7:
            DCR = np.dot(C[5:7, :].T, p.flatten()[5:7])
        elif (C.shape[0] == 5) & (self.noParallaxFit == 1):
            DCR = (np.array(C[4, :]) * p[4]).flatten()
        elif C.shape[0] == 6:
            DCR = (np.array(C[5, :]) * p[5]).flatten()
        elif C.shape[0] == 9:
            DCR = np.dot(C[7:9, :].T, p.flatten()[7:9])
            ACC = np.dot(C[5:7, :].T, p.flatten()[5:7])
            self.ACC = ACC
        elif (C.shape[0] == 5) & (self.noParallaxFit == 0):
            DCR = np.zeros(len(T['da_mas']))

        self.DCR = DCR
        self.ppm_meas = self.T['da_mas'] - self.DCR

        if self.psi_deg is not None:
            # compute epoch averages
            medi = np.unique(T['OB'])
            self.medi = medi
            self.t_MJD_epoch = np.zeros(len(medi))
            self.stdResidualX = np.zeros(len(medi))
            self.errResidualX = np.zeros(len(medi))
            self.Xmean = np.zeros(len(medi))
            self.parfXmean = np.zeros(len(medi))
            self.DCR_Xmean = np.zeros(len(medi))
            self.ACC_Xmean = np.zeros(len(medi))
            self.meanResidualX = np.zeros(len(medi))
            self.x_e_laz = np.zeros(len(medi))
            self.sx_star_laz = np.zeros(len(medi))

            for jj, epoch in enumerate(self.medi):
                tmpidx = np.where(self.T['OB'] == epoch)[0]
                tmpIndexX = tmpidx
                self.t_MJD_epoch[jj] = np.mean(self.T['MJD'][tmpIndexX])
                self.Xmean[jj] = np.average(self.ppm_meas[tmpIndexX],
                                            weights=1. / (self.T['sigma_da_mas'][tmpIndexX] ** 2.))
                self.DCR_Xmean[jj] = np.average(self.DCR[tmpIndexX])
                self.meanResidualX[jj] = np.average(omc[tmpIndexX],
                                                    weights=1. / (self.T['sigma_da_mas'][tmpIndexX] ** 2.))
                self.parfXmean[jj] = np.average(self.T['ppfact'][tmpIndexX])
                self.stdResidualX[jj] = np.std(omc[tmpIndexX])
                if len(tmpIndexX) == 1:
                    self.stdResidualX[jj] = self.T['sigma_da_mas'][tmpIndexX]
                self.errResidualX[jj] = self.stdResidualX[jj] / np.sqrt(len(tmpIndexX))

                # %         from Lazorenko writeup:
                self.x_e_laz[jj] = np.sum(omc[tmpIndexX] / (self.T['sigma_da_mas'][tmpIndexX] ** 2.)) / np.sum(
                    1 / (self.T['sigma_da_mas'][tmpIndexX] ** 2.))
                self.sx_star_laz[jj] = 1 / np.sqrt(np.sum(1 / (self.T['sigma_da_mas'][tmpIndexX] ** 2.)))

            self.chi2_naive = np.sum([self.meanResidualX ** 2 / self.errResidualX ** 2])
            self.chi2_laz = np.sum([self.x_e_laz ** 2 / self.errResidualX ** 2])
            self.chi2_star_laz = np.sum([self.x_e_laz ** 2 / self.sx_star_laz ** 2])
            self.nFree_ep = len(medi) * 2 - C.shape[0]

            self.chi2_laz_red = self.chi2_laz / self.nFree_ep
            self.chi2_star_laz_red = self.chi2_star_laz / self.nFree_ep
            self.chi2_naive_red = self.chi2_naive / self.nFree_ep

            self.epoch_omc_std_X = np.std(self.meanResidualX)
            self.epoch_omc_std = np.std([self.meanResidualX])

        else:

            # compute epoch averages
            medi = np.unique(T['OB'])
            self.medi = medi
            self.t_MJD_epoch = np.zeros(len(medi))
            self.stdResidualX = np.zeros(len(medi))
            self.stdResidualY = np.zeros(len(medi))
            self.errResidualX = np.zeros(len(medi))
            self.errResidualY = np.zeros(len(medi))
            self.Xmean = np.zeros(len(medi))
            self.Ymean = np.zeros(len(medi))
            self.parfXmean = np.zeros(len(medi))
            self.parfYmean = np.zeros(len(medi))
            self.DCR_Xmean = np.zeros(len(medi))
            self.DCR_Ymean = np.zeros(len(medi))
            self.ACC_Xmean = np.zeros(len(medi))
            self.ACC_Ymean = np.zeros(len(medi))
            self.meanResidualX = np.zeros(len(medi))
            self.meanResidualY = np.zeros(len(medi))
            self.x_e_laz = np.zeros(len(medi))
            self.y_e_laz = np.zeros(len(medi))
            self.sx_star_laz = np.zeros(len(medi))
            self.sy_star_laz = np.zeros(len(medi))

            outlier_1D_index = np.array([])
            # loop through epochs
            for jj, epoch in enumerate(self.medi):
                tmpidx = np.where(self.T['OB'] == epoch)[0]
                tmpIndexX = np.intersect1d(self.xi, tmpidx)
                tmpIndexY = np.intersect1d(self.yi, tmpidx)

                self.t_MJD_epoch[jj] = np.mean(self.T['MJD'][tmpIndexX])
                #             print 'epoch %1.0f' % epoch
                #             print self.T['MJD'][tmpIndexX]
                #             pdb.set_trace()

                #             print jj,tmpIndexX
                self.Xmean[jj] = np.average(self.ppm_meas[tmpIndexX],
                                            weights=1. / (self.T['sigma_da_mas'][tmpIndexX] ** 2.))
                self.Ymean[jj] = np.average(self.ppm_meas[tmpIndexY],
                                            weights=1. / (self.T['sigma_da_mas'][tmpIndexY] ** 2.))
                # pdb.set_trace()
                self.DCR_Xmean[jj] = np.average(self.DCR[tmpIndexX])
                self.DCR_Ymean[jj] = np.average(self.DCR[tmpIndexY])
                try:
                    self.ACC_Xmean[jj] = np.average(self.ACC[tmpIndexX])
                    self.ACC_Ymean[jj] = np.average(self.ACC[tmpIndexY])
                except AttributeError:
                    pass

                    #             pdb.set_trace()
                self.meanResidualX[jj] = np.average(omc[tmpIndexX],
                                                    weights=1. / (self.T['sigma_da_mas'][tmpIndexX] ** 2.))
                self.meanResidualY[jj] = np.average(omc[tmpIndexY],
                                                    weights=1. / (self.T['sigma_da_mas'][tmpIndexY] ** 2.))

                self.parfXmean[jj] = np.average(self.T['ppfact'][tmpIndexX])
                self.parfYmean[jj] = np.average(self.T['ppfact'][tmpIndexY])

                self.stdResidualX[jj] = np.std(omc[tmpIndexX])
                self.stdResidualY[jj] = np.std(omc[tmpIndexY])

                if absolute_threshold is not None:
                    outliers_x = (np.abs(omc[tmpIndexX] - np.mean(omc[tmpIndexX])) > outlier_sigma_threshold * self.stdResidualX[jj]) | (np.abs(omc[tmpIndexX] - np.mean(omc[tmpIndexX])) > absolute_threshold)

                    outliers_y = (np.abs(omc[tmpIndexY] - np.mean(omc[tmpIndexY])) > outlier_sigma_threshold * self.stdResidualY[jj]) | (np.abs(omc[tmpIndexY] - np.mean(omc[tmpIndexY])) > absolute_threshold)

                else:
                    outliers_x = np.abs(omc[tmpIndexX] - np.mean(omc[tmpIndexX])) > outlier_sigma_threshold * \
                                                                                self.stdResidualX[jj]
                    outliers_y = np.abs(omc[tmpIndexY] - np.mean(omc[tmpIndexY])) > outlier_sigma_threshold * \
                                                                                    self.stdResidualY[jj]
                if any(outliers_x):
                    tmp_1D_index_x = np.where(outliers_x)[0]
                    print('Detected %d X-residual outliers (%2.1f sigma) in epoch %d (1-indexed) ' % (
                    len(tmp_1D_index_x), outlier_sigma_threshold, epoch), end='')
                    print(np.abs(omc[tmpIndexX] - np.mean(omc[tmpIndexX]))[tmp_1D_index_x], end='')
                    for ii in tmp_1D_index_x:
                        print(' {:.12f}'.format(self.T['MJD'][tmpIndexX[ii]]), end=',')
                    print()

                    outlier_1D_index = np.hstack((outlier_1D_index, tmpIndexX[tmp_1D_index_x]))
                    # outlier_1D_index.append(tmpIndexX[tmp_1D_index_x].tolist())

                if any(outliers_y):
                    tmp_1D_index_y = np.where(outliers_y)[0]
                    print('Detected %d Y-residual outliers (%2.1f sigma) in epoch %d (1-indexed) ' % (
                    len(tmp_1D_index_y), outlier_sigma_threshold, epoch), end='')
                    print(np.abs(omc[tmpIndexY] - np.mean(omc[tmpIndexY]))[tmp_1D_index_y], end='')
                    for ii in tmp_1D_index_y:
                        print(' {:.12f}'.format(self.T['MJD'][tmpIndexY[ii]]), end=',')
                    print()
                    outlier_1D_index = np.hstack((outlier_1D_index, tmpIndexX[tmp_1D_index_y]))
                    # outlier_1D_index.append(tmpIndexY[tmp_1D_index_y].tolist())

                if len(tmpIndexX) == 1:
                    self.stdResidualX[jj] = self.T['sigma_da_mas'][tmpIndexX]
                if len(tmpIndexY) == 1:
                    self.stdResidualY[jj] = self.T['sigma_da_mas'][tmpIndexY]

                self.errResidualX[jj] = self.stdResidualX[jj] / np.sqrt(len(tmpIndexX))
                self.errResidualY[jj] = self.stdResidualY[jj] / np.sqrt(len(tmpIndexY))

                # %         from Lazorenko writeup:
                self.x_e_laz[jj] = np.sum(omc[tmpIndexX] / (self.T['sigma_da_mas'][tmpIndexX] ** 2.)) / np.sum(
                    1 / (self.T['sigma_da_mas'][tmpIndexX] ** 2.))
                self.y_e_laz[jj] = np.sum(omc[tmpIndexY] / (self.T['sigma_da_mas'][tmpIndexY] ** 2.)) / np.sum(
                    1 / (self.T['sigma_da_mas'][tmpIndexY] ** 2.))

                self.sx_star_laz[jj] = 1 / np.sqrt(np.sum(1 / (self.T['sigma_da_mas'][tmpIndexX] ** 2.)))
                self.sy_star_laz[jj] = 1 / np.sqrt(np.sum(1 / (self.T['sigma_da_mas'][tmpIndexY] ** 2.)))

            if len(outlier_1D_index) != 0:
                print('MJD of outliers:')
                for ii in np.unique(outlier_1D_index.astype(np.int)):
                    print('{:.12f}'.format(self.T['MJD'][ii]), end=',')
                print()

                # print(np.unique(self.T['MJD'][outlier_1D_index.astype(np.int)].data))
            # write outliers to file
            if epoch_outlier_dir is not None:
                out_file = os.path.join(epoch_outlier_dir, 'epoch_1D_outliers.txt')

                # T = Table([outlier_1D_index.astype(np.int)], names=['index_1D'])
                # write outlier epoch to file
                T = Table([self.T['MJD'][outlier_1D_index.astype(np.int)]], names=['MJD_1D'])
                T.write(out_file, format='ascii.basic')

            self.outlier_1D_index = outlier_1D_index

            self.chi2_naive = np.sum(
                [self.meanResidualX ** 2 / self.errResidualX ** 2, self.meanResidualY ** 2 / self.errResidualY ** 2])
            self.chi2_laz = np.sum(
                [self.x_e_laz ** 2 / self.errResidualX ** 2, self.y_e_laz ** 2 / self.errResidualY ** 2])
            self.chi2_star_laz = np.sum(
                [self.x_e_laz ** 2 / self.sx_star_laz ** 2, self.y_e_laz ** 2 / self.sy_star_laz ** 2])
            self.nFree_ep = len(medi) * 2 - C.shape[0]

            self.chi2_laz_red = self.chi2_laz / self.nFree_ep
            self.chi2_star_laz_red = self.chi2_star_laz / self.nFree_ep
            self.chi2_naive_red = self.chi2_naive / self.nFree_ep

            self.epoch_omc_std_X = np.std(self.meanResidualX)
            self.epoch_omc_std_Y = np.std(self.meanResidualY)
            self.epoch_omc_std = np.std([self.meanResidualX, self.meanResidualY])


    def ppm_plot(self, save_plot=0, plot_dir=None, name_seed='', descr=None, omc2D=0, arrowOffsetX=0, arrowOffsetY=0,
                 horizons_file_seed=None, psi_deg=None, instrument=None, separate_residual_panels=0,
                 residual_y_axis_limit=None, individual_frame_figure=False, omc_description=None):
        """Make figures showing results of PPM fitting.

        Parameters
        ----------
        save_plot
        plot_dir
        name_seed
        descr
        omc2D
        arrowOffsetX
        arrowOffsetY
        horizons_file_seed
        psi_deg
        instrument
        separate_residual_panels
        residual_y_axis_limit
        individual_frame_figure

        """

        if self.noParallaxFit != 1:
            # orb = OrbitSystem(P_day=1., ecc=0.0, m1_MS=1.0, m2_MJ=0.0, omega_deg=0., OMEGA_deg=0., i_deg=0., Tp_day=0,
            #                   RA_deg=self.RA_deg, DE_deg=self.DE_deg, plx_mas=self.p[2], muRA_mas=self.p[3],
            #                   muDE_mas=self.p[4], Tref_MJD=self.tref_MJD)
            argument_dict = {'m2_MJ'   : 0, 'RA_deg': self.RA_deg, 'DE_deg': self.DE_deg,
                             'absolute_plx_mas' : self.p[2], 'muRA_mas': self.p[3], 'muDE_mas': self.p[4],
                             'Tref_MJD': self.tref_MJD, }
            orb = OrbitSystem(argument_dict)

        else:
            orb = OrbitSystem(P_day=1., ecc=0.0, m1_MS=1.0, m2_MJ=0.0, omega_deg=0., OMEGA_deg=0., i_deg=0., Tp_day=0,
                              RA_deg=self.RA_deg, DE_deg=self.DE_deg, plx_mas=0, muRA_mas=self.p[2],
                              muDE_mas=self.p[3])

        if separate_residual_panels:
            n_subplots = 3
        else:
            n_subplots = 2

        ##################################################################
        # Figure with on-sky motion only, showing individual frames
        if individual_frame_figure:
            fig = pl.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
            pl.clf()

            if instrument is None:
                if psi_deg is None:
                    ppm_curve = orb.ppm(self.tmodel_MJD, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                                        horizons_file_seed=horizons_file_seed, psi_deg=psi_deg)
                ppm_meas = orb.ppm(self.t_MJD_epoch, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                                   horizons_file_seed=horizons_file_seed, psi_deg=psi_deg)
                if psi_deg is None:
                    pl.plot(ppm_curve[0], ppm_curve[1], 'k-')
                    pl.plot(self.Xmean, self.Ymean, 'ko')
                    pl.plot(self.ppm_meas[self.xi], self.ppm_meas[self.yi], 'b.')

            pl.axis('equal')
            ax = plt.gca()
            ax.invert_xaxis()
            pl.xlabel('Offset in Right Ascension (mas)')
            pl.ylabel('Offset in Declination (mas)')
            if self.title is not None:
                pl.title(self.title)
            if save_plot:
                fig_name = os.path.join(plot_dir, 'PPM_{}_frames.pdf'.format(name_seed.replace('.', 'p')))
                plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

        ##################################################################
        # Figure with on-sky motion and residuals
        fig = pl.figure(figsize=(6, 8), facecolor='w', edgecolor='k')
        pl.clf()
        pl.subplot(n_subplots, 1, 1)

        if instrument is None:
            if psi_deg is None:
                ppm_curve = orb.ppm(self.tmodel_MJD, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1], horizons_file_seed=horizons_file_seed, psi_deg=psi_deg)
            # ppm_meas = orb.ppm(self.t_MJD_epoch, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1], horizons_file_seed=horizons_file_seed, psi_deg=psi_deg)
            if psi_deg is None:
                pl.plot(ppm_curve[0], ppm_curve[1], 'k-')
                pl.plot(self.Xmean, self.Ymean, 'ko')

        else:
            instr = np.unique(instrument)
            myColours = np.array(['k', 'b', 'g', '0.7', 'g'])
            for jjj, ins in enumerate(instr):
                tmpInstrument = np.array([ins] * len(self.tmodel_MJD))
                idx = np.where(instrument == ins)[0]
                if psi_deg is None:
                    ppm_curve = orb.ppm(self.tmodel_MJD, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                                        instrument=tmpInstrument, psi_deg=psi_deg)
                    pl.plot(ppm_curve[0], ppm_curve[1], c=myColours[jjj], ls='-')
                    pl.plot(self.Xmean[idx], self.Ymean[idx], marker='o', mfc=myColours[jjj], mec=myColours[jjj],
                            ls='None')
            ppm_meas = orb.ppm(self.t_MJD_epoch, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                               instrument=instrument, psi_deg=psi_deg)



            # arrowOffsetY = 0.
        #         plt.annotate('', xy=(self.p[3][0], self.p[4][0]+arrowOffsetY), xytext=(0, 0+arrowOffsetY), arrowprops=dict(arrowstyle="->",facecolor='black'), size=30 )
        plt.annotate('', xy=(np.float(self.p[3]) + arrowOffsetX, np.float(self.p[4]) + arrowOffsetY),
                     xytext=(0. + arrowOffsetX, 0. + arrowOffsetY), arrowprops=dict(arrowstyle="->", facecolor='black'),
                     size=30)

        pl.axis('equal')
        ax = plt.gca()
        ax.invert_xaxis()
        pl.xlabel('Offset in Right Ascension (mas)')
        pl.ylabel('Offset in Declination (mas)')
        if self.title is not None:
            pl.title(self.title)

        if descr is not None:
            pl.text(0.01, 0.99, descr, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

        pl.subplot(n_subplots, 1, 2)
        epochTime = self.t_MJD_epoch - self.tref_MJD
        epochOrdinateLabel = 'MJD - %3.1f' % self.tref_MJD

        pl.plot(epochTime, self.meanResidualX, 'ko', color='0.7', label='RA')
        pl.errorbar(epochTime, self.meanResidualX, yerr=self.errResidualX, fmt='none', ecolor='0.7')
        plt.axhline(y=0, color='0.5', ls='--', zorder=-50)
        pl.ylabel('O-C (mas)')
        if residual_y_axis_limit is not None:
            pl.ylim((-residual_y_axis_limit, residual_y_axis_limit))
        if psi_deg is None:
            if separate_residual_panels:
                pl.subplot(n_subplots, 1, 3)

            pl.plot(epochTime, self.meanResidualY, 'ko', label='Dec')
            pl.errorbar(epochTime, self.meanResidualY, yerr=self.errResidualY, fmt='none', ecolor='k')
            plt.axhline(y=0, color='0.5', ls='--', zorder=-50)
            pl.ylabel('O-C (mas)')
            if residual_y_axis_limit is not None:
                pl.ylim((-residual_y_axis_limit, residual_y_axis_limit))
        if not separate_residual_panels:
            # pl.legend(loc='best')
            pl.legend(loc=3)

        if omc_description is not None:
            ax=pl.gca()
            pl.text(0.01, 0.99, omc_description, horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes)

        if instrument is not None:
            for jjj, ins in enumerate(instr):
                idx = np.where(instrument == ins)[0]
                pl.plot(epochTime[idx], self.meanResidualY[idx], marker='o', mfc=myColours[jjj], mec=myColours[jjj],
                        ls='None', label=ins)
            pl.legend(loc='best')

        pl.xlabel(epochOrdinateLabel)
        fig.tight_layout(h_pad=0.0)
        pl.show()

        if save_plot:
            fig_name = os.path.join(plot_dir, 'PPM_%s.pdf' % (name_seed.replace('.', 'p')))
            plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)


        if self.C.shape[0] > 7:
            pl.figure(figsize=(6, 8), facecolor='w', edgecolor='k')
            pl.clf()
            pl.subplot(2, 1, 1)
            # pl.plot(self.Xmean - ppm_meas[0],self.Ymean-ppm_meas[1],'ko')
            pl.plot(self.ACC_Xmean, self.ACC_Ymean, 'ko')
            pl.axis('equal')
            ax = plt.gca()
            ax.invert_xaxis()
            pl.xlabel('Offset in Right Ascension (mas)')
            pl.ylabel('Offset in Declination (mas)')
            pl.title('Acceleration')
            pl.subplot(2, 1, 2)
            pl.plot(self.t_MJD_epoch, self.ACC_Xmean, 'ko', color='0.7')
            pl.plot(self.t_MJD_epoch, self.ACC_Ymean, 'ko')
            pl.xlabel('MJD')
            pl.show()

            if save_plot:
                fig_name = os.path.join(plot_dir, 'ACCEL_%s.pdf' % (name_seed.replace('.', 'p')))
                plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

        if omc2D == 1:
            pl.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
            pl.clf()
            pl.plot(self.meanResidualX, self.meanResidualY, 'ko')
            pl.errorbar(self.meanResidualX, self.meanResidualY, xerr=self.errResidualX, yerr=self.errResidualY,
                        fmt='none', ecolor='k')
            pl.axis('equal')
            ax = plt.gca()
            ax.invert_xaxis()
            pl.xlabel('Residual in Right Ascension (mas)')
            pl.ylabel('Residual in Declination (mas)')
            pl.show()
            if save_plot:
                fig_name = '%sPPM_omc2D_%s.pdf' % (plot_dir, name_seed.replace('.', 'p'))
                plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

        elif omc2D == 2:  # for LUH16 referee

            pl.figure(figsize=(6, 8), facecolor='w', edgecolor='k')
            pl.clf()
            pl.subplot(3, 1, 1)
            pl.plot(epochTime, self.Xmean, 'ko', color='0.7')
            pl.plot(epochTime, self.Ymean, 'ko')
            pl.subplot(3, 1, 2)
            pl.ylabel('Offset in RA/Dec (mas)')

            pl.subplot(3, 1, 2)
            pl.plot(self.T['MJD'][self.xi] - self.tref_MJD, self.omc[self.xi], 'ko', color='0.7')
            pl.plot(self.T['MJD'][self.yi] - self.tref_MJD, self.omc[self.yi], 'ko')
            pl.ylabel('Frame O-C (mas)')

            pl.subplot(3, 1, 3)
            # epochOrdinateLabel = 'MJD - %3.1f' % self.tref_MJD
            pl.plot(epochTime, self.meanResidualX, 'ko', color='0.7')
            pl.errorbar(epochTime, self.meanResidualX, yerr=self.errResidualX, fmt='none', ecolor='0.7')
            pl.plot(epochTime, self.meanResidualY, 'ko')
            pl.errorbar(epochTime, self.meanResidualY, yerr=self.errResidualY, fmt='none', ecolor='k')
            plt.axhline(y=0, color='0.5', ls='--', zorder=-50)

            pl.ylabel('Epoch O-C (mas)')
            pl.xlabel(epochOrdinateLabel)
            pl.show()

            if save_plot:
                fig_name = os.path.join(plot_dir, 'PPM_%s_referee.pdf' % (name_seed.replace('.', 'p')))
                plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

    def print_residual_stats(self):
        print('Epoch residual RMS X %3.3f mas' % (self.epoch_omc_std_X))
        if self.psi_deg is None:
            print('Epoch residual RMS Y %3.3f mas' % (self.epoch_omc_std_Y))
        print('Epoch residual RMS   %3.3f mas' % (self.epoch_omc_std))
        print('Degrees of freedom %d' % (self.nFree_ep))
        for elm in ['chi2_laz_red', 'chi2_star_laz_red', 'chi2_naive_red']:
            print('reduced chi^2 : %3.2f (%s)' % (eval('self.%s' % elm), elm))
        if self.psi_deg is None:
            print('Epoch   precision (naive)'),
            print((np.mean([self.errResidualX, self.errResidualY], axis=0)))
            print('Epoch   precision (x_e_laz)'),
            print((np.mean([self.sx_star_laz, self.sy_star_laz], axis=0)))
            print('Average precision (naive) %3.3f mas' % (np.mean([self.errResidualX, self.errResidualY])))
            print('Average precision (x_e_laz) %3.3f mas' % (np.mean([self.sx_star_laz, self.sy_star_laz])))
        else:
            print('Epoch   precision (naive)', )
            print((np.mean([self.errResidualX], axis=0)))
            print('Epoch   precision (x_e_laz)'),
            print((np.mean([self.sx_star_laz], axis=0)))
            print('Average precision (naive) %3.3f mas' % (np.mean([self.errResidualX])))
            print('Average precision (x_e_laz) %3.3f mas' % (np.mean([self.sx_star_laz])))


class AstrometricOrbitPlotter(object):
    """Class to plot results of astrometric fitting of parallax + proper motion + orbit.

    That is, this class supports primarly plotting of barycentric and photocentric orbits.

    Attributes
    ----------
    p : array
        holding best fit parameters of linear fit (usually positions,parallax,proper motion)
        part of what linfit returns
    C : matrix
        Numpy Matrix holding the parameters of the linear model

    Methods
    -------
    """

    def __init__(self, attribute_dict=None):
        """
        theta, C, T, xi, yi, Tref_MJD, omc=None, m1_MS=1.0, outlier_sigma_threshold=3., absolute_threshold=10,
        Parameters
        ----------
        theta : list
            list of dictionaries, length = nr of companions
        C
        T
        xi
        yi
        Tref_MJD
        omc
        m1_MS
        outlier_sigma_threshold
        absolute_threshold
        attribute_dict
        """

        # model_parameters dict (theta)
        # linear_coefficients dict ('matrix', 'table')
        # 2d_indices dict 'xi', 'yi'
        # data_type str '1d', '2d', 'mixed'




        if attribute_dict is not None:
            for key, value in attribute_dict.items():
                setattr(self, key, value)

            # set defaults
            default_dict = {'outlier_sigma_threshold': 3.,
                            'absolute_threshold': 10.,
                            'residuals': None,
                            'scan_angle_definition': 'hipparcos',
                            'include_ppm': True,
                            'title': None,
                            'relative_orbit': False,
                            'verbose': False,
                            }

            for key, value in default_dict.items():
                if key not in attribute_dict.keys():
                    setattr(self, key, value)

        required_attributes = ['linear_coefficients', 'model_parameters', 'data']
        for attribute_name in required_attributes:
            if hasattr(self, attribute_name) is False:
                raise ValueError('Instance has to have a attribute named: {}'.format(attribute_name))


        self.attribute_dict = attribute_dict
        linear_coefficient_matrix = self.linear_coefficients['matrix']

        number_of_companions = len(self.model_parameters)

        self.number_of_companions = number_of_companions
        model_name = 'k{:d}'.format(number_of_companions)


        if self.relative_orbit:
            # assert hasattr(self, 'relative_astrometry')
            assert self.relative_coordinate_system is not None

        T = self.data.epoch_data

        # parameters of first companion
        theta_0 = self.model_parameters[0]
        required_parameters = ['offset_alphastar_mas', 'offset_delta_mas', 'absolute_plx_mas',
                               'muRA_mas', 'muDE_mas']
        theta_names = theta_0.keys()
        for parameter_name in required_parameters:
            if parameter_name not in theta_names:
                raise ValueError('Model parameter {} has to be set!'.format(parameter_name))


        # if ('plx_abs_mas' in theta_names) & ('plx_corr_mas' in theta_names):
        #     theta_0['plx_mas']= theta_0['plx_abs_mas'] + ['plx_corr_mas']

        if 'parallax_correction_mas' in theta_names:
            parallax_for_ppm_mas = theta_0['absolute_plx_mas'] - theta_0['parallax_correction_mas']
        else:
            parallax_for_ppm_mas = theta_0['absolute_plx_mas']

        # compute positions at measurement dates according to best-fit model p (no dcr)
        ppm_parameters = np.array([theta_0['offset_alphastar_mas'], theta_0['offset_delta_mas'],
                                   parallax_for_ppm_mas, theta_0['muRA_mas'], theta_0['muDE_mas']])

        if self.include_ppm:
            self.ppm_model = np.array(np.dot(linear_coefficient_matrix[0:len(ppm_parameters), :].T, ppm_parameters)).flatten()
        elif self.relative_orbit:
            self.ppm_model = np.zeros(len(T))
        else:
            # these are only the positional offsets
            self.ppm_model = np.array(np.dot(linear_coefficient_matrix[0:2, :].T, ppm_parameters[0:2])).flatten()

        if ('esinw' in theta_names):
            # self.ecc, self.omega_deg = mcmc_helpers.decode_eccentricity_omega(theta_0['esinw'], theta_0['ecosw'])
            for p in range(number_of_companions):
                self.model_parameters[p]['ecc'], self.model_parameters[p]['omega_deg'] = \
                    mcmc_helpers.decode_eccentricity_omega(self.model_parameters[p]['esinw'], self.model_parameters[p]['ecosw'])
        if ('m2sini' in theta_names):
            for p in range(number_of_companions):
                self.model_parameters[p]['m2_MJ'], self.model_parameters[p]['i_deg'] = \
                    mcmc_helpers.decode_eccentricity_omega(self.model_parameters[p]['m2sini'], self.model_parameters[p]['m2cosi'])


        if 'rho_mas' in theta_names:
            if 'd_mas' in theta_names:
                dcr_parameters = np.array([theta_0['rho_mas'], theta_0['d_mas']])
            else:
                dcr_parameters = np.array([theta_0['rho_mas']])

            # compute measured positions (dcr-corrected)
            if linear_coefficient_matrix.shape[0] == 7:
                dcr = np.dot(linear_coefficient_matrix[5:7, :].T, dcr_parameters)
            elif linear_coefficient_matrix.shape[0] == 6:
                dcr = linear_coefficient_matrix[5, :] * dcr_parameters
            elif linear_coefficient_matrix.shape[0] <= 5:
                dcr = np.zeros(linear_coefficient_matrix.shape[1])
        else:
            dcr = np.zeros(linear_coefficient_matrix.shape[1])
        self.DCR = dcr

        for p in range(number_of_companions):
            theta_p = self.model_parameters[p]
            if 'm2_MS' in theta_names:
                theta_p['m2_MJ'] = theta_p['m2_MS'] * MS_kg / MJ_kg

            tmporb = OrbitSystem(attribute_dict=theta_p)
            if self.relative_orbit:
                orbit_model = tmporb.relative_orbit_fast(np.array(T['MJD']), np.array(T['spsi']),
                                                         np.array(T['cpsi']),
                                                         shift_omega_by_pi=True,
                                                         coordinate_system=self.relative_coordinate_system)

            else:
                orbit_model = tmporb.photocenter_orbit(np.array(T['MJD']),np.array(T['spsi']),
                                                                          np.array(T['cpsi']))
                # orbit_model = tmporb.pjGetBarycentricAstrometricOrbitFast(np.array(T['MJD']),
                #                                                           np.array(T['spsi']),
                #                                                           np.array(T['cpsi']))

            setattr(self, 'orbit_system_companion_{:d}'.format(p), tmporb)
            setattr(self, 'orbit_model_%d' % (p), orbit_model)

        if number_of_companions == 1:
            self.orbit_system = self.orbit_system_companion_0
            self.orbit_model = self.orbit_model_0
        else:
            self.orbit_model = self.orbit_model_0 + self.orbit_model_1

        if self.residuals is None:
            residuals = np.array(T['da_mas']) - self.orbit_model - self.DCR - self.ppm_model
        else:
            residuals = self.residuals

        if np.any(np.isnan(residuals)):
            raise ValueError('NaN found in residuals')

        self.ppm_meas = np.array(T['da_mas']) - self.DCR - self.orbit_model
        self.orb_meas = np.array(T['da_mas']) - self.DCR - self.ppm_model

        for p in range(number_of_companions):
            if number_of_companions == 1:
                tmp_orb_meas = self.orb_meas
            elif p == 0:
                tmp_orb_meas = np.array(T['da_mas']) - self.DCR - self.ppm_model - self.orbit_model_1
            elif p == 1:
                tmp_orb_meas = np.array(T['da_mas']) - self.DCR - self.ppm_model - self.orbit_model_0
            setattr(self, 'orb_{:d}_meas'.format(p), tmp_orb_meas)

        # compute epoch averages
        medi = np.unique(T['OB'])
        self.medi = medi
        self.n_epoch = len(self.medi)
        self.t_MJD_epoch = np.zeros(self.n_epoch)

        average_quantities_1d = 'stdResidualX errResidualX Xmean_ppm Xmean_orb parfXmean ' \
                                'DCR_Xmean ACC_Xmean meanResidualX x_e_laz sx_star_laz mean_cpsi mean_spsi'.split()

        for p in range(number_of_companions):
            average_quantities_1d += ['Xmean_orb_{:d}'.format(p)]

        for attribute in average_quantities_1d:
            setattr(self, attribute, np.zeros(len(medi)))
        if '2d' in self.data_type:
            for attribute in average_quantities_1d:
                setattr(self, attribute.replace('X', 'Y').replace('x_', 'y_'), np.zeros(len(medi)))

        outlier_1D_index = np.array([])

        if self.data_type == 'gaia_2d':
            self.xi = self.data.xi
            self.yi = self.data.yi

        for jj, epoch in enumerate(self.medi):
            tmpidx = np.where(T['OB'] == epoch)[0]

            if '2d' in self.data_type:
                tmpIndexX = np.intersect1d(self.xi, tmpidx)
                tmpIndexY = np.intersect1d(self.yi, tmpidx)
            elif self.data_type == '1d':
                tmpIndexX = tmpidx

            self.t_MJD_epoch[jj] = np.mean(T['MJD'][tmpIndexX])
            self.mean_cpsi[jj] = np.mean(T['cpsi'][tmpIndexX])
            self.mean_spsi[jj] = np.mean(T['spsi'][tmpIndexX])

            self.Xmean_ppm[jj] = np.average(self.ppm_meas[tmpIndexX],
                                            weights=1. / (np.array(T['sigma_da_mas'])[tmpIndexX] ** 2.))
            self.Xmean_orb[jj] = np.average(self.orb_meas[tmpIndexX],
                                            weights=1. / (T['sigma_da_mas'][tmpIndexX] ** 2.))

            if np.any(np.isnan(self.Xmean_ppm)):
                raise ValueError('NaN found in Xmean_ppm')
            if np.any(np.isnan(self.Xmean_orb)):
                raise ValueError('NaN found in Xmean_orb')

            if '2d' in self.data_type:

                self.Ymean_ppm[jj] = np.average(self.ppm_meas[tmpIndexY],
                                                weights=1. / (T['sigma_da_mas'][tmpIndexY] ** 2.))
                self.Ymean_orb[jj] = np.average(self.orb_meas[tmpIndexY],
                                                weights=1. / (T['sigma_da_mas'][tmpIndexY] ** 2.))

            for p in range(number_of_companions):
                getattr(self, 'Xmean_orb_{:d}'.format(p))[jj] = np.average(
                    getattr(self, 'orb_{:d}_meas'.format(p))[tmpIndexX],
                    weights=1. / (T['sigma_da_mas'][tmpIndexX] ** 2.))
                # if self.data_type == '2d':
                if '2d' in self.data_type:
                    getattr(self, 'Ymean_orb_{:d}'.format(p))[jj] = np.average(
                        getattr(self, 'orb_{:d}_meas'.format(p))[tmpIndexY],
                        weights=1. / (T['sigma_da_mas'][tmpIndexY] ** 2.))

            self.DCR_Xmean[jj] = np.average(self.DCR[tmpIndexX])
            self.meanResidualX[jj] = np.average(residuals[tmpIndexX], weights=1. / (T['sigma_da_mas'][tmpIndexX] ** 2.))
            self.parfXmean[jj] = np.average(T['ppfact'][tmpIndexX])
            self.stdResidualX[jj] = np.std(residuals[tmpIndexX]) if len(tmpIndexX)>1 else T['sigma_da_mas'][tmpIndexX]


            if '2d' in self.data_type:
                self.DCR_Ymean[jj] = np.average(self.DCR[tmpIndexY])
                self.meanResidualY[jj] = np.average(residuals[tmpIndexY], weights=1. / (T['sigma_da_mas'][tmpIndexY] ** 2.))
                self.parfYmean[jj] = np.average(T['ppfact'][tmpIndexY])
                self.stdResidualY[jj] = np.std(residuals[tmpIndexY]) if len(tmpIndexY)>1 else T['sigma_da_mas'][tmpIndexY]

            # on the fly inter-epoch outlier detection
            outliers = {}
            outliers['x'] = {}
            outliers['x']['index'] = tmpIndexX
            outliers['x']['std_residual'] = self.stdResidualX[jj]

            if '2d' in self.data_type:
                outliers['y'] = {}
                outliers['y']['index'] = tmpIndexY
                outliers['y']['std_residual'] = self.stdResidualY[jj]

            is_outlier = []
            for key in outliers.keys():
                # boolean array
                if self.absolute_threshold is not None:
                    is_outlier = (np.abs(residuals[outliers[key]['index']] - np.mean(residuals[outliers[key]['index']])) > self.outlier_sigma_threshold * outliers[key]['std_residual']) | (
                             np.abs(residuals[outliers[key]['index']] - np.mean(residuals[outliers[key]['index']])) > self.absolute_threshold)

                elif self.outlier_sigma_threshold is not None:
                    is_outlier = np.abs(residuals[outliers[key]['index']] - np.mean(residuals[outliers[key]['index']])) > self.outlier_sigma_threshold * outliers[key]['std_residual']

                if any(is_outlier):
                    tmp_1D_index = np.where(is_outlier)[0]
                    print('Detected {} {}-residual outliers ({:2.1f} sigma) in epoch {} (1-indexed) '.format(
                            len(tmp_1D_index), key, self.outlier_sigma_threshold, epoch), end='')
                    print(np.abs(residuals[outliers[key]['index']] - np.mean(residuals[outliers[key]['index']]))[tmp_1D_index], end='')
                    # 1/0
                    for ii in tmp_1D_index:
                        print(' {:.12f}'.format(T['MJD'][outliers[key]['index'][ii]]), end=',')
                    print()

                    outlier_1D_index = np.hstack((outlier_1D_index, outliers[key]['index'][tmp_1D_index]))


            self.errResidualX[jj] = self.stdResidualX[jj] / np.sqrt(len(tmpIndexX))

            if '2d' in self.data_type:
                self.errResidualY[jj] = self.stdResidualY[jj] / np.sqrt(len(tmpIndexY))

            # %         from Lazorenko writeup:
            self.x_e_laz[jj] = np.sum(residuals[tmpIndexX] / (T['sigma_da_mas'][tmpIndexX] ** 2.)) / np.sum(
                1 / (T['sigma_da_mas'][tmpIndexX] ** 2.))
            self.sx_star_laz[jj] = 1 / np.sqrt(np.sum(1 / (T['sigma_da_mas'][tmpIndexX] ** 2.)));

            if '2d' in self.data_type:
                self.y_e_laz[jj] = np.sum(residuals[tmpIndexY] / (T['sigma_da_mas'][tmpIndexY] ** 2.)) / np.sum(
                    1 / (T['sigma_da_mas'][tmpIndexY] ** 2.))
                self.sy_star_laz[jj] = 1 / np.sqrt(np.sum(1 / (T['sigma_da_mas'][tmpIndexY] ** 2.)));

        if len(outlier_1D_index) != 0:
            print('MJD of outliers:')
            for ii in np.unique(outlier_1D_index.astype(np.int)):
                print('{:.12f}'.format(T['MJD'][ii]), end=',')
            print()

        self.outlier_1D_index = np.array(outlier_1D_index).astype(int)

        # compute chi squared values
        if self.data_type == '1d':
            self.chi2_naive = np.sum([self.meanResidualX ** 2 / self.errResidualX ** 2])
            self.chi2_laz = np.sum([self.x_e_laz ** 2 / self.errResidualX ** 2])
            self.chi2_star_laz = np.sum([self.x_e_laz ** 2 / self.sx_star_laz ** 2])
        elif '2d' in self.data_type:
            self.chi2_naive = np.sum(
                [self.meanResidualX ** 2 / self.errResidualX ** 2, self.meanResidualY ** 2 / self.errResidualY ** 2])
            self.chi2_laz = np.sum(
                [self.x_e_laz ** 2 / self.errResidualX ** 2, self.y_e_laz ** 2 / self.errResidualY ** 2])
            self.chi2_star_laz = np.sum(
                [self.x_e_laz ** 2 / self.sx_star_laz ** 2, self.y_e_laz ** 2 / self.sy_star_laz ** 2])

        # fixed 2018-08-18 JSA
        if self.data_type == '1d':
            self.nFree_ep = len(medi) * 1 - (linear_coefficient_matrix.shape[0] + number_of_companions*7)
        elif '2d' in self.data_type:
                self.nFree_ep = len(medi) * 2 - (linear_coefficient_matrix.shape[0] + number_of_companions*7)

        self.chi2_laz_red = self.chi2_laz / self.nFree_ep
        self.chi2_star_laz_red = self.chi2_star_laz / self.nFree_ep
        self.chi2_naive_red = self.chi2_naive / self.nFree_ep

        self.epoch_omc_std_X = np.std(self.meanResidualX)
        if self.data_type == '1d':
            self.epoch_omc_std = self.epoch_omc_std_X
            self.epoch_precision_mean = np.mean([self.errResidualX])
        elif '2d' in self.data_type:
            self.epoch_omc_std_Y = np.std(self.meanResidualY)
            self.epoch_omc_std = np.std([self.meanResidualX, self.meanResidualY])
            self.epoch_precision_mean = np.mean([self.errResidualX, self.errResidualY])

        self.residuals = residuals


    def epoch_parameters(self):
        """Return structure with epoch mean parameters to facilitate e.g. detection limit computation.

        Returns
        -------

        """

        cat = Table()
        cat['MJD'] = self.t_MJD_epoch
        cat['RA*_mas'] = self.Xmean_ppm
        cat['DE_mas'] = self.Ymean_ppm
        cat['sRA*_mas'] = self.errResidualX
        cat['sDE_mas'] = self.errResidualY
        cat['OB'] = self.medi
        cat['frame'] = self.medi

        iad = ImagingAstrometryData(cat, data_type=self.data_type)
        iad.RA_deg = self.orbit_system.RA_deg
        iad.Dec_deg = self.orbit_system.DE_deg
        iad.set_five_parameter_coefficients()
        iad.set_data_1D()

        # covariance matrix
        S_mean = np.mat(np.diag(1. / np.power(iad.data_1D['sigma_da_mas'], 2)))

        # mean signal/abscissa
        M_mean = np.mat(iad.data_1D['da_mas'])

        # coefficient matrix
        C_mean = iad.five_parameter_coefficients_array

        mean_dict = {'covariance_matrix': S_mean,
                     'signal': M_mean,
                     'coefficient_matrix': C_mean,
                     'iad': iad
                     }

        # return C_mean, S_mean, M_mean
        return mean_dict


    def print_residual_statistics(self):
        """Print statistics to stdout."""
        print('='*100)
        print('Epoch residual RMS X %3.3f mas' % (self.epoch_omc_std_X))
        if self.data_type == '2d':
            print('Epoch residual RMS Y %3.3f mas' % (self.epoch_omc_std_Y))
        print('Epoch residual RMS   %3.3f mas' % (self.epoch_omc_std))
        print('Degrees of freedom %d' % (self.nFree_ep))
        for elm in ['chi2_laz_red', 'chi2_star_laz_red', 'chi2_naive_red']:
            print('reduced chi^2 : %3.2f (%s)' % (eval('self.%s' % elm), elm))
        print('Epoch   precision (naive)'),
        print(self.epoch_precision_mean)
        if self.data_type == '1d':
            print('Epoch   precision (x_e_laz)'),
            print(np.mean([self.sx_star_laz], axis=0))
            print('Average precision (naive) %3.3f mas' % (np.mean([self.errResidualX])))
            print('Average precision (x_e_laz) %3.3f mas' % (np.mean([self.sx_star_laz])))
        elif '2d' in self.data_type:
            print('Epoch   precision (x_e_laz)'),
            print(np.mean([self.sx_star_laz, self.sy_star_laz], axis=0))
            print('Average precision (naive) %3.3f mas' % (np.mean([self.errResidualX, self.errResidualY])))
            print('Average precision (x_e_laz) %3.3f mas' % (np.mean([self.sx_star_laz, self.sy_star_laz])))
        print('='*100)


    def astrometric_signal_to_noise_epoch(self, amplitude_mas):
        """Return astrometric SNR for epochs (FOV transists not CCD transits)"""
        if self.data_type == '1d':
            median_uncertainty_mas = np.median([self.errResidualX])
        astrometric_snr = amplitude_mas * np.sqrt(self.n_epoch)/median_uncertainty_mas
        return astrometric_snr

    def plot(self, argument_dict=None):
        """Make the astrometric orbit plots.

        Parameters
        ----------
        argument_dict : dict

        """
        # set defaults
        if argument_dict is not None:
            default_argument_dict = {'arrow_length_factor': 1.,
                            'horizons_file_seed': None,
                            'frame_omc_description': 'default',
                            'orbit_description': 'default',
                            'scan_angle_definition': 'gaia',
                            'orbit_signal_description': 'default',
                            'ppm_description': 'default',
                            'epoch_omc_description': 'default',
                            'name_seed': 'star',
                            'make_1d_overview_figure': True,
                            'make_condensed_summary_figure': True,
                            'frame_residual_panel': False,
                            'arrow_offset_x': 40.,
                            'arrow_offset_y': 0.,
                            'save_plot': False,
                            'orbit_only_panel': False,
                            'make_xy_residual_figure': False,
                            'make_ppm_figure': False,
                            'plot_dir': os.getcwd(),
                            }

            for key, value in default_argument_dict.items():
                if key not in argument_dict.keys():
                    argument_dict[key] = value

        if argument_dict['ppm_description'] == 'default':
            argument_dict['ppm_description'] = '$\\varpi={:2.3f}$ mas\n$\mu_\\mathrm{{ra^\\star}}={' \
                        ':2.3f}$ mas/yr\n$\mu_\\mathrm{{dec}}={:2.3f}$ mas/yr'.format(
                self.model_parameters[0]['absolute_plx_mas'], self.model_parameters[0]['muRA_mas'],
                self.model_parameters[0]['muDE_mas'])

        if argument_dict['epoch_omc_description'] == 'default':
            argument_dict['epoch_omc_description'] = '$N_e={}$, $N_f={}$,\n$\Delta t={:.0f}$ d, DOF$_\\mathrm{{eff}}$={},\n' \
                              '$\Sigma_\\mathrm{{O-C,epoch}}$={:2.3f} mas\n$\\bar\\sigma_\Lambda$={:2.3f} mas'.format(
                len(np.unique(self.data.epoch_data['OB'])), len(self.data.epoch_data),
                np.ptp(self.data.epoch_data['MJD']), self.nFree_ep, self.epoch_omc_std,
                self.epoch_precision_mean)

        if argument_dict['frame_omc_description'] == 'default':
            argument_dict['frame_omc_description'] = '$N_f={}$, $\Sigma_\\mathrm{{O-C,frame}}$={:2.3f} mas\n' \
                                    '$\\bar\\sigma_\Lambda$={:2.3f} mas'.format(
                len(self.data.epoch_data), np.std(self.residuals), np.mean(self.data.epoch_data['sigma_da_mas']))
            if 'excess_noise' in argument_dict.keys():
                argument_dict['frame_omc_description'] += '\nexN = {:2.2f}, mF = {:2.0f}'.format(
            argument_dict['excess_noise'], argument_dict['merit_function'])

        if argument_dict['orbit_signal_description'] == 'default':
            argument_dict[
                'orbit_signal_description'] = '$\Sigma_\\mathrm{{Signal,epoch}}$={:2.3f} mas'.format(
                np.std(self.Xmean_orb))

        #  loop over number of companions
        for p in range(self.number_of_companions):
            if argument_dict['orbit_description'] == 'default':
                argument_dict['tmp_orbit_description'] = '$P={:2.3f}$ d\n$e={:2.3f}$\n$\\alpha={:2.3f}$ mas\n$i={:2.3f}$ deg\n$\\omega={:2.3f}$ deg\n$\\Omega={:2.3f}$ deg\n$M_1={:2.3f}$ Msun\n$M_2={:2.1f}$ Mjup'.format(self.model_parameters[p]['P_day'], self.model_parameters[p]['ecc'], getattr(self, 'orbit_system_companion_{:d}'.format(p)).alpha_mas, self.model_parameters[p]['i_deg'], self.model_parameters[p]['omega_deg'], self.model_parameters[p]['OMEGA_deg'], self.model_parameters[p]['m1_MS'], self.model_parameters[p]['m2_MJ'])
            else:
                argument_dict['tmp_orbit_description'] = argument_dict['orbit_description']


            theta_p = self.model_parameters[p]
            theta_names = theta_p.keys()
            name_seed_2 = argument_dict['name_seed'] + '_companion{:d}'.format(p)

            if 'm2_MS' in theta_names:
                theta_p['m2_MJ'] = theta_p['m2_MS'] * MS_kg / MJ_kg
            # if ('plx_abs_mas' in theta_names) & ('plx_corr_mas' in theta_names):
            #     theta_p['plx_mas'] = theta_p['plx_abs_mas'] + theta_p['plx_corr_mas']

            orb = OrbitSystem(attribute_dict=theta_p)
            if getattr(orb, 'Tref_MJD') is None:
                raise UserWarning('Reference time was not set.')

            # PPM plot and residuals
            if argument_dict['make_ppm_figure']:
                n_rows = 2
                n_columns = 1
                fig = pl.figure(figsize=(6, 8), facecolor='w', edgecolor='k')
                pl.clf()

                # PPM panel
                pl.subplot(n_rows, n_columns, 1)
                self.insert_ppm_plot(orb, argument_dict)

                pl.axis('equal')
                ax = plt.gca()
                ax.invert_xaxis()
                pl.xlabel('Offset in Right Ascension (mas)')
                pl.ylabel('Offset in Declination (mas)')
                if self.title is not None:
                    pl.title(self.title)

                pl.subplot(n_rows, n_columns, 2)
                self.insert_epoch_residual_plot(orb, argument_dict)

                plt.tight_layout()
                pl.show()
                if argument_dict['save_plot']:
                    figure_file_name = os.path.join(argument_dict['plot_dir'],
                                                        'ppm_{}.pdf'.format(
                                                            name_seed_2.replace('.', 'p')))
                    plt.savefig(figure_file_name, transparent=True, bbox_inches='tight',
                                pad_inches=0.05)


            # 1D astrometry overview figure
            if argument_dict['make_1d_overview_figure']:
                n_rows = 3
                n_columns = 2
                fig = pl.figure(figsize=(14, 9), facecolor='w', edgecolor='k')
                pl.clf()

                # PPM panel
                pl.subplot(n_rows, n_columns, 1)
                self.insert_ppm_plot(orb, argument_dict)
                pl.axis('equal')
                ax = plt.gca()
                ax.invert_xaxis()
                pl.xlabel('Offset in Right Ascension (mas)')
                pl.ylabel('Offset in Declination (mas)')
                if self.title is not None:
                    pl.title(self.title)

                # orbit panel
                pl.subplot(n_rows-1, n_columns, 3)
                self.insert_orbit_plot(orb, argument_dict)
                pl.axis('equal')
                ax = plt.gca()
                ax.invert_xaxis()
                pl.xlabel('Offset in Right Ascension (mas)')
                pl.ylabel('Offset in Declination (mas)')

                pl.subplot(n_rows, n_columns, 2)
                self.insert_orbit_timeseries_plot(orb, argument_dict)
                pl.subplot(n_rows, n_columns, 4)
                self.insert_orbit_epoch_residuals_plot(orb, argument_dict)
                pl.subplot(n_rows, n_columns, 6)
                self.insert_orbit_frame_residuals_plot(orb, argument_dict, direction='x')
                pl.xlabel('MJD - {:3.1f}'.format(orb.Tref_MJD))

                # fig.tight_layout(h_pad=0.0)
                pl.show()
                if argument_dict['save_plot']:
                    figure_file_name = os.path.join(argument_dict['plot_dir'],
                                                        'orbit_1d_summary_{}.pdf'.format(
                                                            name_seed_2.replace('.', 'p')))
                    plt.savefig(figure_file_name, transparent=True, bbox_inches='tight',
                                pad_inches=0.05)

            ##################################################
            # TRIPLE PANEL FIGURE (PPM + ORBIT + EPOCH RESIDUALS)
            # plot PPM and residuals
            if argument_dict['make_condensed_summary_figure']:
                if argument_dict['frame_residual_panel']:
                    pl.figure(figsize=(6, 9), facecolor='w', edgecolor='k')
                    n_panels = 3
                else:
                    pl.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
                    n_panels = 2
                pl.clf()

                # PPM panel
                pl.subplot(n_panels, 1, 1)
                self.insert_ppm_plot(orb, argument_dict)
                pl.axis('equal')
                ax = plt.gca()
                ax.invert_xaxis()
                pl.xlabel('Offset in Right Ascension (mas)')
                pl.ylabel('Offset in Declination (mas)')
                if self.title is not None:
                    pl.title(self.title)

                # orbit panel
                pl.subplot(n_panels, 1, 2)
                self.insert_orbit_plot(orb, argument_dict)
                pl.axis('equal')
                ax = plt.gca()
                ax.invert_xaxis()
                pl.xlabel('Offset in Right Ascension (mas)')
                pl.ylabel('Offset in Declination (mas)')

                # frame residual panel
                if argument_dict['frame_residual_panel']:
                    pl.subplot(n_panels, 1, 3)
                    self.insert_epoch_residual_plot(orb, argument_dict)

                plt.tight_layout()
                pl.show()

                if argument_dict['save_plot']:
                    figure_file_name = os.path.join(argument_dict['plot_dir'], 'ppm_orbit_{}.pdf'.format(name_seed_2.replace('.', 'p')))
                    plt.savefig(figure_file_name, transparent=True, bbox_inches='tight', pad_inches=0.05)
                ##################################################


            ##################################################
            #  ORBIT only
            if argument_dict['orbit_only_panel']:
                pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
                pl.clf()

                self.insert_orbit_plot(orb, argument_dict)
                if self.title is not None:
                    pl.title(self.title)
                pl.axis('equal')
                ax = plt.gca()
                ax.invert_xaxis()
                pl.xlabel('Offset in Right Ascension (mas)')
                pl.ylabel('Offset in Declination (mas)')
                pl.show()
                if argument_dict['save_plot']:
                    figure_file_name = os.path.join(argument_dict['plot_dir'], 'orbit_only_{}.pdf'.format(name_seed_2.replace('.', 'p')))
                    plt.savefig(figure_file_name, transparent=True, bbox_inches='tight', pad_inches=0.05)
            ##################################################


            ##################################################
            # FIGURE SHOWING RA AND Dec OFFSETS AND RESIDUALS
            if argument_dict['make_xy_residual_figure']:
                if self.data_type == '1d':
                    n_columns = 1
                elif self.data_type == '2d':
                    n_columns = 2

                if argument_dict['frame_residual_panel']:
                    n_rows = 3
                elif argument_dict['omc_panel'] is False:
                    n_rows = 1
                else:
                    n_rows = 2

                fig, axes = pl.subplots(n_rows, n_columns, sharex=True, sharey=False, figsize=(n_columns*4.0, n_rows*2.5), facecolor='w',
                                        edgecolor='k', squeeze=False)

                self.insert_orbit_timeseries_plot(orb, argument_dict, ax=axes[0][0])
                if self.data_type == '2d':
                    self.insert_orbit_timeseries_plot(orb, argument_dict, direction='y', ax=axes[0][1])
                    if self.title is not None:
                        fig.suptitle(self.title)

                if argument_dict['omc_panel']:
                    self.insert_orbit_epoch_residuals_plot(orb, argument_dict, ax=axes[1][0])
                    if self.data_type == '2d':
                        self.insert_orbit_epoch_residuals_plot(orb, argument_dict, direction='y', ax=axes[1][1])

                if argument_dict['frame_residual_panel']:
                    self.insert_orbit_frame_residuals_plot(orb, argument_dict, direction='x', ax=axes[2][0])
                    if self.data_type == '2d':
                        self.insert_orbit_frame_residuals_plot(orb, argument_dict, direction='y',
                                                               ax=axes[2][1])
                axes[-1][0].set_xlabel('MJD - %3.1f' % orb.Tref_MJD)
                labels = axes[-1][0].get_xticklabels()
                plt.setp(labels, rotation=30)
                if self.data_type == '2d':
                    axes[-1][1].set_xlabel('MJD - %3.1f' % orb.Tref_MJD)
                    labels = axes[-1][1].get_xticklabels()
                    plt.setp(labels, rotation=30)

                # if self.title is None:
                #     fig.tight_layout(pad=0.0)
                # plt.tight_layout()
                # pl.subplots_adjust(right=1.5)
                pl.show()
                if argument_dict['save_plot']:
                    if argument_dict['frame_residual_panel']:
                        figure_file_name = os.path.join(argument_dict['plot_dir'], 'orbit_time_{}_frameres.pdf'.format(name_seed_2.replace('.', 'p')))
                    else:
                        figure_file_name = os.path.join(argument_dict['plot_dir'],
                                               'orbit_time_{}.pdf'.format(name_seed_2.replace('.', 'p')))
                    plt.savefig(figure_file_name, transparent=True, bbox_inches='tight', pad_inches=0.05)


            # if argument_dict['make_relative_orbit_figure']:



    def insert_ppm_plot(self, orb, argument_dict):
        """Plot the PPM model curve and the orbit-substracted, epoch-averaged measurements.

        Parameters
        ----------
        orb
        argument_dict

        Returns
        -------

        """

        t_curve_mjd_2d = np.sort(np.tile(self.t_curve_MJD, 2))

        ppm_curve = orb.ppm(t_curve_mjd_2d, offsetRA_mas=orb.offset_alphastar_mas,
                            offsetDE_mas=orb.offset_delta_mas,
                            horizons_file_seed=argument_dict['horizons_file_seed'])

        pl.plot(ppm_curve[0], ppm_curve[1], 'k-')
        if self.data_type == '2d':
            pl.plot(self.Xmean_ppm, self.Ymean_ppm, 'ko')
        plt.annotate('', xy=(np.float(orb.muRA_mas) * argument_dict['arrow_length_factor'] + argument_dict['arrow_offset_x'],
                             np.float(orb.muDE_mas) * argument_dict['arrow_length_factor'] + argument_dict['arrow_offset_y']),
                     xytext=(0. + argument_dict['arrow_offset_x'], 0. + argument_dict['arrow_offset_y']),
                     arrowprops=dict(arrowstyle="->", facecolor='black'), size=30)

        if argument_dict['ppm_description'] is not None:
            ax = pl.gca()
            pl.text(0.01, 0.99, argument_dict['ppm_description'], horizontalalignment='left',
                    verticalalignment='top', transform=ax.transAxes)


    def insert_orbit_timeseries_plot(self, orb, argument_dict, direction='x', ax=None):
        """Plot the residual signal after removal of parallax, proper motion, linear terms."""

        if ax is None:
            ax = pl.gca()

        ax.axhline(y=0, color='0.5', ls=':', zorder=-50)

        if direction=='x':
            ax.plot(self.t_MJD_epoch - orb.Tref_MJD, self.Xmean_orb, 'ko')
            ax.errorbar(self.t_MJD_epoch - orb.Tref_MJD, self.Xmean_orb, yerr=self.errResidualX,
                                fmt='none', ecolor='k')
            if argument_dict['orbit_signal_description'] is not None:
                pl.text(0.01, 0.99, argument_dict['orbit_signal_description'], horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes)

        if self.data_type == '1d':
            ax.set_ylabel('Offset along scan (mas)')
            # ax.set_title(self.title)

        elif self.data_type == '2d':
            timestamps_1D, cpsi_curve, spsi_curve, xi_curve, yi_curve = get_cpsi_spsi_for_2Dastrometry(
                self.t_curve_MJD, scan_angle_definition=argument_dict['scan_angle_definition'])
            # orbit_curve = orb.pjGetBarycentricAstrometricOrbitFast(timestamps_1D, spsi_curve,
            #                                                        cpsi_curve)
            if self.relative_orbit:
                orbit_curve = orb.relative_orbit_fast(timestamps_1D, spsi_curve, cpsi_curve,
                                                      shift_omega_by_pi=True,
                                                      coordinate_system=self.relative_coordinate_system)
            else:
                orbit_curve = orb.photocenter_orbit(timestamps_1D, spsi_curve,
                                                                   cpsi_curve)
            phi1_curve = orbit_curve[xi_curve]
            phi2_curve = orbit_curve[yi_curve]

            if direction=='x':
                ax.plot(self.t_curve_MJD - orb.Tref_MJD, phi1_curve, 'k-')
                ax.set_ylabel('Offset in RA/Dec (mas)')
            elif direction=='y':
                ax.plot(self.t_MJD_epoch - orb.Tref_MJD, self.Ymean_orb, 'ko')
                ax.errorbar(self.t_MJD_epoch - orb.Tref_MJD, self.Ymean_orb, yerr=self.errResidualY,
                            fmt='none', ecolor='k')
                ax.plot(self.t_curve_MJD - orb.Tref_MJD, phi2_curve, 'k-')
                # ax.set_ylabel('Offset in Dec (mas)')



    def insert_orbit_epoch_residuals_plot(self, orb, argument_dict, direction='x', ax=None):
        """

        Parameters
        ----------
        orb
        argument_dict
        direction
        ax

        Returns
        -------

        """

        if ax is None:
            ax = pl.gca()

        if direction=='x':
            ax.plot(self.t_MJD_epoch - orb.Tref_MJD, self.meanResidualX, 'ko')
            ax.errorbar(self.t_MJD_epoch - orb.Tref_MJD, self.meanResidualX,
                                yerr=self.errResidualX, fmt='none', ecolor='k')
            ax.axhline(y=0, color='0.5', ls='--', zorder=-50)
            ax.set_ylabel('O-C (mas)')
            if argument_dict['epoch_omc_description'] is not None:
                pl.text(0.01, 0.99, argument_dict['epoch_omc_description'], horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes)

        elif direction=='y':
            ax.plot(self.t_MJD_epoch - orb.Tref_MJD, self.meanResidualY, 'ko')
            ax.errorbar(self.t_MJD_epoch - orb.Tref_MJD, self.meanResidualY,
                                yerr=self.errResidualY, fmt='none', ecolor='k')
            ax.axhline(y=0, color='0.5', ls='--', zorder=-50)
            # ax.set_ylabel('O-C (mas)')

    def insert_orbit_frame_residuals_plot(self, orb, argument_dict, direction='x', ax=None):
        """

        Parameters
        ----------
        orb
        argument_dict
        direction
        ax

        Returns
        -------

        """

        if ax is None:
            ax = pl.gca()

        if self.data_type == '1d':
            ax.plot(self.data.epoch_data['MJD'] - orb.Tref_MJD, self.residuals, 'ko', mfc='k', ms=4)
            ax.errorbar(self.data.epoch_data['MJD'] - orb.Tref_MJD, self.residuals, yerr=self.data.epoch_data['sigma_da_mas'], fmt='none', ecolor='k')
            ax.axhline(y=0, color='0.5', ls='--', zorder=-50)

            # 1/0
            if len(self.outlier_1D_index) != 0:
                ax.plot(self.data.epoch_data['MJD'][self.outlier_1D_index] - orb.Tref_MJD, self.residuals[self.outlier_1D_index], 'ko', mfc='b',
                        ms=4)
                # 1/0
                ax.errorbar(np.array(self.data.epoch_data['MJD'])[self.outlier_1D_index] - orb.Tref_MJD, self.residuals[self.outlier_1D_index],
                            yerr=np.array(self.data.epoch_data['sigma_da_mas'])[self.outlier_1D_index], fmt='none', ecolor='b')

            if argument_dict['frame_omc_description'] is not None:
                pl.text(0.01, 0.99, argument_dict['frame_omc_description'], horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes)

        elif self.data_type == '2d':

            if direction=='x':
                tmp_index =  self.xi
            elif direction=='y':
                tmp_index = self.yi

            # mfc = 'none'
            mec= '0.4'
            mfc = mec
            marker='.'
            alpha = 0.5
            ax.plot(self.data.epoch_data['MJD'][tmp_index] - orb.Tref_MJD, self.residuals[tmp_index], mec=mec, mfc=mfc, marker=marker, ls='none', alpha=alpha)
            ax.axhline(y=0, color='0.5', ls='--', zorder=-50)

        ax.set_ylabel('Frame O-C (mas)')


    def insert_epoch_residual_plot(self, orb, argument_dict):
        """Plot the epoch-average residuals.

        Parameters
        ----------
        orb
        argument_dict

        Returns
        -------

        """

        epochTime = self.t_MJD_epoch - orb.Tref_MJD
        epochOrdinateLabel = 'MJD - {:3.1f}'.format(orb.Tref_MJD)
        if self.data_type == '2d':
            x_residual_color = '0.7'
        else:
            x_residual_color = 'k'
        pl.plot(epochTime, self.meanResidualX, 'ko', color=x_residual_color)
        pl.errorbar(epochTime, self.meanResidualX, yerr=self.errResidualX, fmt='none',
                    ecolor=x_residual_color)
        if self.data_type == '2d':
            pl.plot(epochTime, self.meanResidualY, 'ko')
            pl.errorbar(epochTime, self.meanResidualY, yerr=self.errResidualY, fmt='none', ecolor='k')
        plt.axhline(y=0, color='0.5', ls='--', zorder=-50)

        pl.ylabel('O-C (mas)')
        pl.xlabel(epochOrdinateLabel)
        if argument_dict['epoch_omc_description'] is not None:
            ax = plt.gca()
            pl.text(0.01, 0.99, argument_dict['epoch_omc_description'], horizontalalignment='left',
                    verticalalignment='top', transform=ax.transAxes)

    def insert_orbit_plot(self, orb, argument_dict):
        """Add orbit to current figure.

        Returns
        -------

        """

        timestamps_1D, cpsi_curve, spsi_curve, xi_curve, yi_curve = get_cpsi_spsi_for_2Dastrometry(self.t_curve_MJD, scan_angle_definition=argument_dict['scan_angle_definition'])
        if self.relative_orbit:
            orbit_curve = orb.relative_orbit_fast(timestamps_1D, spsi_curve, cpsi_curve, shift_omega_by_pi=True,
                                                     coordinate_system=self.relative_coordinate_system)
        else:
            orbit_curve = orb.photocenter_orbit(timestamps_1D, spsi_curve, cpsi_curve)
            # orbit_curve = orb.pjGetBarycentricAstrometricOrbitFast(timestamps_1D, spsi_curve, cpsi_curve)
        phi1_curve = orbit_curve[xi_curve]
        phi2_curve = orbit_curve[yi_curve]

        t_epoch_MJD, cpsi_epoch, spsi_epoch, xi_epoch, yi_epoch = get_cpsi_spsi_for_2Dastrometry(self.t_MJD_epoch, scan_angle_definition=argument_dict['scan_angle_definition'])
        if self.relative_orbit:
            orbit_epoch = orb.relative_orbit_fast(t_epoch_MJD, spsi_epoch, cpsi_epoch, shift_omega_by_pi=True,
                                                     coordinate_system=self.relative_coordinate_system)
        else:
            orbit_epoch = orb.photocenter_orbit(t_epoch_MJD, spsi_epoch, cpsi_epoch)
            # orbit_epoch = orb.pjGetBarycentricAstrometricOrbitFast(t_epoch_MJD, spsi_epoch, cpsi_epoch)
        phi1_model_epoch = orbit_epoch[xi_epoch]
        phi2_model_epoch = orbit_epoch[yi_epoch]

        t_frame_mjd, cpsi_frame, spsi_frame, xi_frame, yi_frame = get_cpsi_spsi_for_2Dastrometry(np.array(self.data.epoch_data['MJD']), scan_angle_definition=argument_dict['scan_angle_definition'])
        if self.relative_orbit:
            orbit_frame = orb.relative_orbit_fast(t_frame_mjd, spsi_frame, cpsi_frame, shift_omega_by_pi=True,
                                                     coordinate_system=self.relative_coordinate_system)
        else:
            orbit_frame = orb.photocenter_orbit(t_frame_mjd, spsi_frame, cpsi_frame)
            # orbit_frame = orb.pjGetBarycentricAstrometricOrbitFast(t_frame_mjd, spsi_frame, cpsi_frame)
        phi1_model_frame = orbit_frame[xi_frame]
        phi2_model_frame = orbit_frame[yi_frame]

        # show periastron
        if 1:
            t_periastron_mjd, cpsi_periastron, spsi_periastron, xi_periastron, yi_periastron = get_cpsi_spsi_for_2Dastrometry(orb.Tp_day, scan_angle_definition=argument_dict['scan_angle_definition'])
            if self.relative_orbit:
                orbit_periastron = orb.relative_orbit_fast(t_periastron_mjd, spsi_periastron, cpsi_periastron,
                                                      shift_omega_by_pi=True,
                                                      coordinate_system=self.relative_coordinate_system)
            else:
                orbit_periastron = orb.photocenter_orbit(t_periastron_mjd, spsi_periastron, cpsi_periastron)
                # orbit_periastron = orb.pjGetBarycentricAstrometricOrbitFast(t_periastron_mjd, spsi_periastron, cpsi_periastron)
            phi1_model_periastron = orbit_periastron[xi_periastron]
            phi2_model_periastron = orbit_periastron[yi_periastron]
            pl.plot([0, phi1_model_periastron], [0, phi2_model_periastron], 'k.-', lw=0.5, color='0.5')
            pl.plot(phi1_model_periastron, phi2_model_periastron, 'ks', color='0.5', mfc='0.5')



        pl.plot(phi1_curve, phi2_curve, 'k-', lw=1.5, color='0.5')
        pl.plot(phi1_model_epoch, phi2_model_epoch, 'ko', color='0.7', ms=5, mfc='none')


        if self.data_type in ['1d', 'gaia_2d']:
            # if self.data_type == '1d':
            #     frame_index = np.arange(len(self.residuals))
            # else:
            #     1/0
            if argument_dict['scan_angle_definition'] == 'hipparcos':
                frame_residual_alphastar_along_scan = self.data.epoch_data['cpsi'] * self.residuals
                frame_residual_delta_along_scan = self.data.epoch_data['spsi'] * self.residuals
                epoch_residual_alphastar_along_scan = self.mean_cpsi * self.meanResidualX
                epoch_residual_delta_along_scan = self.mean_spsi * self.meanResidualX
            elif argument_dict['scan_angle_definition'] == 'gaia':
                frame_residual_alphastar_along_scan = self.data.epoch_data['spsi'] * self.residuals
                frame_residual_delta_along_scan = self.data.epoch_data['cpsi'] * self.residuals
                epoch_residual_alphastar_along_scan = self.mean_spsi * self.meanResidualX
                epoch_residual_delta_along_scan = self.mean_cpsi * self.meanResidualX

            frame_residual_color = '0.8'
            pl.plot(phi1_model_frame + frame_residual_alphastar_along_scan,
                    phi2_model_frame + frame_residual_delta_along_scan, 'ko',
                    color=frame_residual_color, ms=4, mfc=frame_residual_color,
                    mec=frame_residual_color)
            pl.plot(phi1_model_epoch + epoch_residual_alphastar_along_scan,
                    phi2_model_epoch + epoch_residual_delta_along_scan, 'ko', color='k',
                    ms=5)  # , mfc='none', mew=2)

            # plot epoch-level error-bars
            for jj in range(len(self.meanResidualX)):
                if argument_dict['scan_angle_definition'] == 'hipparcos':
                    x1 = phi1_model_epoch[jj] + self.mean_cpsi[jj] * (self.meanResidualX[jj] + self.errResidualX[jj])
                    x2 = phi1_model_epoch[jj] + self.mean_cpsi[jj] * (self.meanResidualX[jj] - self.errResidualX[jj])
                    y1 = phi2_model_epoch[jj] + self.mean_spsi[jj] * (self.meanResidualX[jj] + self.errResidualX[jj])
                    y2 = phi2_model_epoch[jj] + self.mean_spsi[jj] * (self.meanResidualX[jj] - self.errResidualX[jj])
                elif argument_dict['scan_angle_definition'] == 'gaia':
                    x1 = phi1_model_epoch[jj] + self.mean_spsi[jj] * (self.meanResidualX[jj] + self.errResidualX[jj])
                    x2 = phi1_model_epoch[jj] + self.mean_spsi[jj] * (self.meanResidualX[jj] - self.errResidualX[jj])
                    y1 = phi2_model_epoch[jj] + self.mean_cpsi[jj] * (self.meanResidualX[jj] + self.errResidualX[jj])
                    y2 = phi2_model_epoch[jj] + self.mean_cpsi[jj] * (self.meanResidualX[jj] - self.errResidualX[jj])
                pl.plot([x1, x2], [y1, y2], 'k-', lw=1)

                #  from yorick code
                #     // psi is the scan angle from north to east (better, from west to north)
                # // scanning direction
                # dx1_mas = cpsi_obs *  myresidual;//*hd.SRES;
                # dy1_mas = spsi_obs *  myresidual;// *hd.SRES;

        elif self.data_type == '2d':
            pl.plot(self.Xmean_orb, self.Ymean_orb, 'ko', ms=8)
            pl.errorbar(self.Xmean_orb, self.Ymean_orb, xerr=self.errResidualX, yerr=self.errResidualY,
                        fmt='none', ecolor='0.6', zorder=-49)
            for j in range(len(phi1_model_epoch)):
                pl.plot([self.Xmean_orb[j], phi1_model_epoch[j]], [self.Ymean_orb[j], phi2_model_epoch[j]],
                        'k--', color='0.7', zorder=-50)

        # show origin
        pl.plot(0, 0, 'kx')

        if argument_dict['tmp_orbit_description'] is not None:
            pl.text(0.01, 0.99, argument_dict['tmp_orbit_description'], horizontalalignment='left',
                    verticalalignment='top', transform=pl.gca().transAxes)


class DetectionLimit(object):
    """Class to support determination of planet detection limits from astrometry."""


    def __init__(self, attribute_dict={}):
        """The default attribute values are stored in the hardcoded
        dictionary below, which also defines the list of acceptable
        attributes.

        The content of attribute_dict is transferred to the instance.

        Parameters
        ----------
        attribute_dict : dict
        """
        self.attribute_dict = attribute_dict
        default_dict = {'m1_msun': 1.,  # primary mass
                        'absolute_plx_mas': 25.,  # parallax
                        'identifier': 'starname',  # name
                        'm2_grid_n': 10,  # number of samples across the secondary mass range
                        'm2_mjup_lower': 1.,  # lower limit for secondary mass
                        'm2_mjup_upper': 30.,  # upper limit for secondary mass
                        'simulations_per_gridpoint_n': 1000,  # number of simulations at any grid point
                        'period_grid_n': 10,  # number of samples across the period range
                        'period_day_lower': 50.,  # lower limit of orbital period
                        'period_day_upper': 1000.,  # lower limit of orbital period
                        'out_dir': os.getcwd(),
                        'overwrite': False
                        }

        # Assign user values as attributes when present, use defaults if not
        attribute_keys = attribute_dict.keys()
        for key, val in default_dict.items():
            if key in attribute_keys:
                setattr(self, key, attribute_dict[key])
            else:
                setattr(self, key, val)

        # Warn users if a key in attribute_dict isn't a default attribute
        mismatch = [key for key in attribute_dict.keys()
                    if key not in default_dict.keys()]
        if mismatch:
            raise KeyError('Key{0} {1} {2} absent in default OrbitClass'
                           .format('s' if len(mismatch) > 1 else '',
                                   mismatch,
                                   'are' if len(mismatch) > 1 else 'is'))

        self.n_simulations = self.period_grid_n* self.simulations_per_gridpoint_n * self.m2_grid_n  # number of planetary systems generated
        print('Instantiating DetectionLimit object:')
        print('Simulations: total number {}: {} periods, {} secondary masses, {} random)'.format(
            self.n_simulations, self.period_grid_n, self.m2_grid_n, self.simulations_per_gridpoint_n))
        print('Simulations: M2 resolution {:3.3f} Mjup'.format((self.m2_mjup_upper - self.m2_mjup_lower) / self.m2_grid_n))

    def prepare_reference_dataset(self, xfP, use_mean_epochs=True, horizonsFileSeed=None):
        """

        Parameters
        ----------
        xfP
        use_mean_epochs
        horizonsFileSeed

        Returns
        -------

        """
        if use_mean_epochs:  # fastSimu works with epoch averages
            # C_mean, S_mean, M_mean = xfP.epoch_parameters()
            mean_parameters = xfP.epoch_parameters()

            res_mean = linearfit.LinearFit(mean_parameters['signal'], mean_parameters['covariance_matrix'],
                                           mean_parameters['coefficient_matrix'])
            res_mean.fit()

            self.S_mean = mean_parameters['covariance_matrix']
            self.C_mean = mean_parameters['coefficient_matrix']
            self.M_mean = mean_parameters['signal']
            self.iad = mean_parameters['iad']
            self.res_mean = res_mean

        # 1/0
        #
        #
        self.tp_mjd = xfP.orbit_system.Tp_day
        #
        #
        #     orb_mean = OrbitSystem(P_day=1., ecc=0.0, m1_MS=1.0, m2_MJ=0.0, omega_deg=0., OMEGA_deg=0., i_deg=0.,
        #                            Tp_day=0, RA_deg=xfP.RA_deg, DE_deg=xfP.DE_deg, plx_mas=self.absPlx_mas, muRA_mas=0,
        #                            muDE_mas=0, Tref_MJD=xfP.tref_MJD)
        #     ppm1dMeas_mean_mas = orb_mean.ppm(xfP.t_MJD_epoch, horizons_file_seed=horizonsFileSeed,
        #                                       psi_deg=xfP.psi_deg)
        #     C_mean = orb_mean.coeffMatrix
        #     TableC1_mean = Table(C_mean.T, names=('cpsi', 'spsi', 'ppfact', 'tcpsi', 'tspsi'))
        #     tmp_mean, xi_mean, yi_mean = xfGetMeanParMatrix(xfP)
        #     S_mean = np.mat(np.diag(1. / np.power(tmp_mean['sigma_da_mas'], 2)))
        #     M_mean = np.mat(tmp_mean['da_mas'])
        #     # res_mean = linfit(M_mean, S_mean, C_mean)
        #     res_mean = linearfit.LinearFit(M_mean, S_mean, C_mean)
        #     res_mean.fit()
        #     # res_mean.makeReadableNumbers()
        #
        #     self.TableC1_mean = TableC1_mean
        #     self.tmp_mean = tmp_mean
        #     self.res_mean = res_mean
        #     self.S_mean = S_mean
        #     self.C_mean = C_mean
        #     # res_mean.disp()

    def run_simulation(self, simu_run=1, log_P_day_grid=True):
        """

        Parameters
        ----------
        simu_run
        log_P_day_grid

        Returns
        -------

        """
        self.m2_jup_grid = np.linspace(self.m2_mjup_lower, self.m2_mjup_upper, self.m2_grid_n)

        if log_P_day_grid:
            self.p_day_grid = np.logspace(np.log10(self.period_day_lower),
                                          np.log10(self.period_day_upper),
                                          self.period_grid_n)
        else:
            self.p_day_grid = np.linspace(self.period_day_lower, self.period_day_upper,
                                          self.period_grid_n)

        simu_dir = os.path.join(self.out_dir, 'simu/simu_run{}/'.format(simu_run))
        if not os.path.exists(simu_dir):
            os.makedirs(simu_dir)

        mc_file_name = os.path.join(simu_dir, '{}_detectionLimits_{}_m1{:1.3f}.pkl'.format(
            self.identifier, self.n_simulations, self.m1_msun))

        mean_residuals = np.zeros((self.n_simulations, len(self.res_mean.residuals)))
        mean_residual_rms = np.zeros(self.n_simulations)

        if ((not os.path.isfile(mc_file_name)) or (self.overwrite)):

            # sample OMEGA space uniformly
            OMEGA_deg_vals = np.linspace(0, 359, 360)
            simu_OMEGA_deg = np.random.choice(OMEGA_deg_vals, self.n_simulations)

            # sample inclination space according to sin(i) probability
            i_deg_vals = np.linspace(0, 179, 180)
            PDF_i_deg = 1. / 2 * np.sin(np.deg2rad(i_deg_vals))
            PDF_i_deg_normed = PDF_i_deg / np.sum(PDF_i_deg)
            simu_i_deg = np.random.choice(i_deg_vals, self.n_simulations, p=PDF_i_deg_normed)

            simu_M2_jup = np.zeros(self.n_simulations)
            temp_M2 = np.zeros(self.m2_grid_n * self.simulations_per_gridpoint_n)
            for jj in range(self.m2_grid_n):
                tempIdx = np.arange(jj * self.simulations_per_gridpoint_n, (jj + 1) * self.simulations_per_gridpoint_n)
                temp_M2[tempIdx] = self.m2_jup_grid[jj] * np.ones(self.simulations_per_gridpoint_n)

            simu_P_day = np.zeros(self.n_simulations)
            for jj in range(self.period_grid_n):
                tempIdx = np.arange(jj * self.simulations_per_gridpoint_n * self.m2_grid_n,
                                    (jj + 1) * self.simulations_per_gridpoint_n * self.m2_grid_n)
                simu_P_day[tempIdx] = self.p_day_grid[jj] * np.ones(self.simulations_per_gridpoint_n * self.m2_grid_n)
                simu_M2_jup[tempIdx] = temp_M2;

            # time of perisatron passage
            simu_tp_mjd = self.tp_mjd + np.random.rand(self.n_simulations) * simu_P_day

            # simulate circular orbits only
            ecc = 0.
            omega_deg = 0.

            if 0:
                pl.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
                pl.clf()
                pl.subplot(2, 2, 1)
                pl.hist(simu_i_deg)
                pl.xlabel('inc')
                pl.subplot(2, 2, 2)
                pl.hist(simu_OMEGA_deg)
                pl.xlabel('OMEGA')
                pl.subplot(2, 2, 3)
                pl.hist(simu_P_day)
                pl.xlabel('Period')
                pl.subplot(2, 2, 4)
                pl.hist(simu_M2_jup)
                pl.xlabel('M2')
                pl.show()

            print('Running simulations ...')
            print('Simulation 0000000')
            spsi = np.array(self.iad.data_1D['spsi'])
            cpsi = np.array(self.iad.data_1D['cpsi'])
            ref_da_mas = np.array(self.M_mean)

            ref_omc_mas = self.res_mean.residuals
            for j in range(self.n_simulations):
                # tot_da_mas = []
                # simu_da_mas = []
                simu_da_mas = pjGetOrbitFast(P_day=simu_P_day[j], ecc=ecc, m1_MS=self.m1_msun, m2_MJ=simu_M2_jup[j],
                                             omega_deg=omega_deg, OMEGA_deg=simu_OMEGA_deg[j], i_deg=simu_i_deg[j],
                                             T0_day=simu_tp_mjd[j], plx_mas=self.absolute_plx_mas,
                                             t_MJD=np.array(self.iad.data_1D['MJD']), spsi=spsi, cpsi=cpsi)
                # orb_simu = OrbitSystem(P_day=simu_P_day[j], ecc=ecc, m1_MS=M1_Msun, m2_MJ = simu_M2_jup[j] , omega_deg=omega_deg, OMEGA_deg=simu_OMEGA_deg[j], i_deg=simu_i_deg[j], Tp_day = simu_tp_mjd[j], RA_deg=RA_deg,DE_deg=DE_deg,plx_mas = plx_mas, muRA_mas=res.p[3][0],muDE_mas=res.p[4][0] )
                # simu_da_mas = orb_simu.pjGetOrbitFast(0 , t_MJD = tmp_mean['MJD'], psi_deg = psi_deg )#, verbose=0):

                tot_da_mas = ref_da_mas - ref_omc_mas + simu_da_mas  # remove noise structure

                simu_res = linearfit.LinearFit(np.mat(tot_da_mas), self.S_mean, self.C_mean)
                simu_res.fit()

                mean_residual_rms[j] = np.std(np.array(simu_res.residuals))
                if np.mod(j, 10000) == 0:
                    print('\b\b\b\b\b\b\b%07d' % j)
                    # print '\x1b[%07d\r' % j,
            pickle.dump((mean_residual_rms), open(mc_file_name, "wb"))

        else:
            mean_residual_rms = pickle.load(open(mc_file_name, "rb"))

        self.mean_residual_rms = mean_residual_rms

    def run_simulation_parallel(self, simulation_run_number=1, log_P_day_grid=True, parallel=True):
        """
        parallelized running of simulations, looping through simulated pseudo-orbits

        :param simulation_run_number:
        :param log_P_day_grid:
        :param parallel:
        :return:
        """

        # directory to write to
        simulation_dir = os.path.join(self.dwDir, 'simulation/simulation_run_number%d/' % simulation_run_number)
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        # generate grid of companion masses
        self.m2_jup_grid = np.linspace(self.m2_mjup_lower, self.m2_mjup_upper, self.m2_grid_n)

        # generate grid of orbital periods (log or linear spacing)
        if log_P_day_grid:
            self.p_day_grid = np.logspace(np.log10(self.period_day_lower), np.log10(self.period_day_upper),
                                          self.period_grid_n)
        else:
            self.p_day_grid = np.linspace(self.period_day_lower, self.period_day_upper, self.period_grid_n)

        # pickle file to save results
        mc_file_name = os.path.join(simulation_dir, 'dw%02d_detectionLimits_%d%s.pkl' % (
            self.dwNr, self.n_simulations, ('_MA%1.3f' % self.M1_Msun).replace('.', 'p')))

        # meanResiduals = np.zeros((self.n_simulations, len(self.res_mean.omc[0])))
        mean_residual_rms = np.zeros(self.n_simulations)

        N_sim_within_loop = self.simulations_per_gridpoint_n * self.m2_grid_n
        # array to hold results, sliced by orbital period
        mean_residual_rms = np.zeros((self.period_grid_n, N_sim_within_loop))

        def compute_mean_residual_rms(P_day, ecc, m1_MS, m2_MJ,
                                      omega_deg, OMEGA_deg, i_deg,
                                      T0_day, plx_mas,
                                      t_MJD, spsi, cpsi, ref_da_mas, ref_omc_mas):

            simu_da_mas = pjGetOrbitFast(P_day, ecc, m1_MS, m2_MJ,
                                         omega_deg, OMEGA_deg, i_deg,
                                         T0_day, plx_mas,
                                         t_MJD, spsi, cpsi)

            tot_da_mas = ref_da_mas - ref_omc_mas + simu_da_mas  # remove noise structure
            simu_res = linfit(np.mat(tot_da_mas), self.S_mean, self.C_mean)
            individual_mean_residual_rms = np.std(np.array(simu_res.omc)[0])

            return individual_mean_residual_rms

        def return_residual_rms_array(arg):
            [P_day, ecc, m1_MS, m2_MJ_array,
             omega_deg, OMEGA_deg_array, i_deg_array,
             T0_day_array, plx_mas,
             t_MJD, spsi, cpsi, ref_da_mas, ref_omc_mas] = arg

            n = len(m2_MJ_array)
            residual_rms_array = np.zeros(n)
            for j in range(n):
                residual_rms_array[j] = compute_mean_residual_rms(P_day, ecc, m1_MS, m2_MJ_array[j],
                                                                  omega_deg, OMEGA_deg_array[j], i_deg_array[j],
                                                                  T0_day_array[j], plx_mas,
                                                                  t_MJD, spsi, cpsi, ref_da_mas, ref_omc_mas)

            return residual_rms_array

        # import numpy as np
        # from multiprocessing import Pool
        from pathos.multiprocessing import ProcessingPool as Pool

        if ((not os.path.isfile(mc_file_name)) or (self.overwrite)):
            random_seed = 1234

            OMEGA_deg_vals = np.linspace(0, 359, 360)
            np.random.seed(random_seed)
            simu_OMEGA_deg = np.random.choice(OMEGA_deg_vals, N_sim_within_loop)

            i_deg_vals = np.linspace(0, 179, 180)
            PDF_i_deg = 1. / 2 * np.sin(np.deg2rad(i_deg_vals))
            PDF_i_deg_normed = PDF_i_deg / np.sum(PDF_i_deg)
            np.random.seed(random_seed)
            simu_i_deg = np.random.choice(i_deg_vals, N_sim_within_loop, p=PDF_i_deg_normed)

            simu_M2_jup = np.zeros(N_sim_within_loop)
            # temp_M2 = np.zeros(self.m2_grid_n * self.simulations_per_gridpoint_n)
            for jj in range(self.m2_grid_n):
                tempIdx = np.arange(jj * self.simulations_per_gridpoint_n, (jj + 1) * self.simulations_per_gridpoint_n)
                simu_M2_jup[tempIdx] = self.m2_jup_grid[jj] * np.ones(self.simulations_per_gridpoint_n)

            # simu_P_day = np.zeros(self.n_simulations)
            # for jj in range(self.period_grid_n):
            #     tempIdx = np.arange(jj * self.simulations_per_gridpoint_n * self.m2_grid_n,
            #                         (jj + 1) * self.simulations_per_gridpoint_n * self.m2_grid_n)
            #     simu_P_day[tempIdx] = self.p_day_grid[jj] * np.ones(self.simulations_per_gridpoint_n * self.m2_grid_n)
            #     simu_M2_jup[tempIdx] = temp_M2;



            ecc = 0.
            omega_deg = 0.

            print('Running simulations in parallel...')
            spsi = np.array(self.TableC1_mean['spsi'])
            cpsi = np.array(self.TableC1_mean['cpsi'])
            ref_da_mas = np.array(self.tmp_mean['da_mas'])
            ref_omc_mas = self.res_mean.omc[0]

            n_processes = 8

            pool = Pool(processes=n_processes)

            arg_list = []
            for jj, P_day in enumerate(self.p_day_grid):
                # print('Processing period number %d'%jj)

                np.random.seed(random_seed)
                simu_T0_day = self.T0_MJD + np.random.rand(N_sim_within_loop) * P_day

                arg = [P_day, ecc, self.M1_Msun, simu_M2_jup,
                       omega_deg, simu_OMEGA_deg, simu_i_deg,
                       simu_T0_day, self.absPlx_mas,
                       np.array(self.tmp_mean['MJD']), spsi, cpsi, ref_da_mas, ref_omc_mas]
                arg_list.append(arg)


            import time

            t0 = time.time()

            mean_residual_rms = np.array(pool.map(return_residual_rms_array, arg_list))
            t1 = time.time()
            print('multiprocessing using %d processes finished in %3.3f sec' % (n_processes, t1 - t0))

            pool.close()

            pickle.dump((mean_residual_rms.flatten()), open(mc_file_name, "wb"))

        else:
            mean_residual_rms = pickle.load(open(mc_file_name, "rb"))

        self.mean_residual_rms = mean_residual_rms.flatten()


    def plot_simu_results(self, xfP, factor=1., visplot=True, confidence_limit=0.997,
                          x_axis_unit='day', semilogx=True, y_data_divisor=None, y_data_factor=1.,
                          new_figure=True, line_width=2.):
        """

        Parameters
        ----------
        xfP
        factor
        visplot
        confidence_limit
        x_axis_unit
        semilogx
        y_data_divisor
        y_data_factor
        new_figure
        line_width

        Returns
        -------

        """

        # if xfP.psi_deg is None:
        if xfP.data_type is '2d':
            criterion = np.std([xfP.meanResidualX, xfP.meanResidualY]) * factor
        else:
            criterion = np.std([xfP.meanResidualX]) * factor
        print('Detection criterion is %3.3f mas ' % (criterion))
        print('Using confidence limit of {:.3f}'.format(confidence_limit))

        n_smaller = np.zeros((self.period_grid_n, self.m2_grid_n))

        for jj in range(self.period_grid_n):
            tempIdx = np.arange(jj * self.simulations_per_gridpoint_n * self.m2_grid_n,
                                (jj + 1) * self.simulations_per_gridpoint_n * self.m2_grid_n)
            for kk in range(self.m2_grid_n):
                pix = np.arange(kk * self.simulations_per_gridpoint_n, (kk + 1) * self.simulations_per_gridpoint_n)
                n_smaller[jj, kk] = np.sum(self.mean_residual_rms[tempIdx[pix]] <= criterion)

        detection_limit = np.zeros((self.period_grid_n, 2))
        for jj in range(self.period_grid_n):
            try:
                limit_index = np.where(n_smaller[jj, :] < self.simulations_per_gridpoint_n * (1 - confidence_limit))[0][0]
                try:
                    M2_val = self.m2_jup_grid[limit_index]
                except ValueError:
                    M2_val = np.max(self.m2_jup_grid)
            except IndexError:
                M2_val = np.max(self.m2_jup_grid)

            detection_limit[jj, :] = [self.p_day_grid[jj], M2_val]

        if visplot:
            if x_axis_unit == 'day':
                x_axis_factor = 1
            elif x_axis_unit == 'year':
                x_axis_factor = 1. / u.year.to(u.day)
            x_axis_label = 'Period ({})'.format(x_axis_unit)

            if new_figure:
                pl.figure(figsize=(6, 3), facecolor='w', edgecolor='k')
                pl.clf()
            if semilogx:
                if y_data_divisor is not None:
                    pl.semilogx(detection_limit[:, 0] * x_axis_factor, y_data_divisor/detection_limit[:, 1]*y_data_factor, 'k-', lw=line_width)
                else:
                    pl.semilogx(detection_limit[:, 0] * x_axis_factor, detection_limit[:, 1]*y_data_factor, 'k-', lw=line_width)
            else:
                if y_data_divisor is not None:
                    pl.plot(detection_limit[:, 0] * x_axis_factor, y_data_divisor/detection_limit[:, 1]*y_data_factor, 'k-', lw=line_width)
                else:
                    pl.plot(detection_limit[:, 0] * x_axis_factor, detection_limit[:, 1] * y_data_factor, 'k-',
                            lw=line_width)
            pl.title('{:.1f}% confidence limit'.format(confidence_limit * 100))
            if y_data_divisor is not None:
                pl.ylim((0, y_data_divisor / np.max(self.m2_jup_grid) * y_data_factor))
            else:
                pl.ylim((0, np.max(self.m2_jup_grid) * y_data_factor))
            pl.xlabel(x_axis_label)
            if new_figure:
                pl.show()

        self.detection_limit = detection_limit


def plot_rv_data(rv, orbit_system=None, verbose=True, n_orbit=2, estimate_systemic_velocity=False,
                 data_colour='k', include_degenerate_orbit=False, plot_parameters_ensemble=None):
    """

    Parameters
    ----------
    rv
    orbit_system
    verbose
    n_orbit
    estimate_systemic_velocity

    Returns
    -------

    """
    rv['jyear'] = [Time(rv['MJD'][i], format='mjd').jyear for i in range(len(rv))]

    n_rows = 2
    n_columns = 1
    fig, axes = pl.subplots(n_rows, n_columns, sharex=True, figsize=(n_rows * 3.5, n_columns * 5.5),
                            facecolor='w', edgecolor='k', squeeze=False)

    if 'rv_mps' in rv.colnames:
        basic_unit = 'mps'
        conversion_factor = 1
    elif 'rv_kmps' in rv.colnames:
        basic_unit = 'kmps'
        conversion_factor = 1e3

    unit_string = {'mps': 'm/s', 'kmps': 'km/s'}

    # fig.suptitle(self.title)

    # pl.subplot(2,1,1)
    axes[0][0].plot(rv['jyear'], rv['rv_{}'.format(basic_unit)], 'ko', label='_', mfc=data_colour)
    axes[0][0].errorbar(rv['jyear'], rv['rv_{}'.format(basic_unit)], yerr=rv['sigma_rv_{}'.format(basic_unit)], fmt='none', ecolor=data_colour, label='_')


    n_rv = len(rv)

    if orbit_system is not None:

        # fit systemic velocity
        if estimate_systemic_velocity:
            rv_mps = orbit_system.compute_radial_velocity(np.array(rv['MJD']))
            rv_kmps = rv_mps / 1000.
            onesvec = np.ones(n_rv)
            C = np.mat([onesvec])
            weight = 1. / np.power(np.array(rv['sigma_rv_kmps']), 2)
            LHS = np.mat(np.array(rv['rv_kmps']) - rv_kmps)
            res = linearfit.LinearFit(LHS, np.diag(weight), C)
            res.fit()
            gamma_kmps = np.float(res.p)
            gamma_mps = gamma_kmps*1e3
            print('Systemic velocity {:2.3f} +/- {:2.3f} km/s'.format(gamma_kmps,
                                                                      res.p_normalised_uncertainty[0]))
            rv['rv_model_kmps'] = rv_kmps + gamma_kmps
            orbit_system.gamma_ms = gamma_mps
        else:
            rv['rv_model_{}'.format(basic_unit)] = orbit_system.compute_radial_velocity(np.array(rv['MJD']))/conversion_factor
            gamma_mps = None
        # plot RV orbit of primary
        time_offset_day = rv['MJD'][0] - orbit_system.Tp_day
        orbit_system.plot_rv_orbit(time_offset_day=time_offset_day, n_orbit=n_orbit,
                                   n_curve=10000, axis=axes[0][0], rv_unit=basic_unit)
        if plot_parameters_ensemble is not None:
            n_curve = 500
            n_ensemble = len(plot_parameters_ensemble['offset_alphastar_mas'])

            # array to store RVs
            rv_ensemble = np.zeros((n_ensemble, n_curve))

            # get times at which to sample orbit
            t_plot_ensemble_jyear = orbit_system.get_t_plot(time_offset_day=time_offset_day, n_orbit=n_orbit, n_curve=n_curve)
            t_plot_ensemble_mjd = orbit_system.get_t_plot(time_offset_day=time_offset_day,
                                                          n_orbit=n_orbit, n_curve=n_curve,
                                                          format='mjd')

            for key in ['m2_MS', 'm_tot_ms', 'P_year', 'a1_mas', 'arel_mas', 'arel_AU']:
                if key in plot_parameters_ensemble.keys():
                    plot_parameters_ensemble.pop(key)
                plot_parameters_ensemble['Tref_MJD'] = np.ones(n_ensemble)*orbit_system.Tref_MJD
            for index_ensemble in range(n_ensemble):
                tmp_system = OrbitSystem({key: samples[index_ensemble] for key, samples in plot_parameters_ensemble.items()})
                rv_ensemble[index_ensemble, :] = tmp_system.compute_radial_velocity(t_plot_ensemble_mjd)/1e3
            axes[0][0].fill_between(t_plot_ensemble_jyear, np.percentile(rv_ensemble, 15.865, axis=0),
                            np.percentile(rv_ensemble, 84.134, axis=0), color='0.7')
            # 1/0
            # orbit_system_ensemble = [OrbitSystem({})]
            # for key,
            # rv_mps = (self.compute_radial_velocity(t_day))

            # 1/0
        if include_degenerate_orbit:
            orbit_system_degenerate = copy.deepcopy(orbit_system)
            orbit_system_degenerate.omega_deg += 180.
            orbit_system_degenerate.OMEGA_deg += 180.
            orbit_system_degenerate.plot_rv_orbit(time_offset_day=rv['MJD'][0] - orbit_system.Tp_day,
                                       n_orbit=n_orbit, n_curve=1000, axis=axes[0][0],
                                       rv_unit=basic_unit, line_style='--')

        residuals = rv['rv_{}'.format(basic_unit)] - rv['rv_model_{}'.format(basic_unit)]
        rv_description = '$\\gamma={:2.3f}$ km/s\n$N_\\mathrm{{RV}}={}$\n' \
                         '$\Sigma_\\mathrm{{O-C}}$={:2.3f} {}'.format(orbit_system.gamma_ms/1e3, len(rv), np.std(residuals), unit_string[basic_unit])

        # plot systemic velocity
        axes[0][0].axhline(y=orbit_system.gamma_ms / conversion_factor, color='0.5', ls=':', zorder=-50)

        axes[1][0].plot(rv['jyear'], residuals, 'ko', label='_', mfc=data_colour)
        axes[1][0].errorbar(rv['jyear'], residuals, yerr=rv['sigma_rv_{}'.format(basic_unit)], fmt='none', ecolor=data_colour, label='_')

        axes[1][0].text(0.01, 0.99, rv_description, horizontalalignment='left',
                verticalalignment='top', transform=axes[1][0].transAxes)

    axes[-1][0].set_xlabel('Time (Julian year)')
    # pl.legend()
    axes[0][0].set_ylabel('RV ({})'.format(unit_string[basic_unit]))
    axes[1][0].set_ylabel('O-C ({})'.format(unit_string[basic_unit]))
    axes[1][0].axhline(y=0, color='0.5', ls='--', zorder=-50)
    axes[1][0].set_xlabel('Time (Julian year)')

    labels = axes[-1][0].get_xticklabels()
    plt.setp(labels, rotation=30)

    fig.tight_layout(h_pad=0.0)

    if verbose:
        rv.pprint()


def get_cpsi_spsi_for_2Dastrometry(timestamps_2D, scan_angle_definition='hipparcos'):
    """Return cos(psi) and sin(psi) for regular 2D astrometry, where psi is the scan angle.

    For Hipparcos
    xi = spsi==0    #index of X coordinates (cpsi = 1) psi =  0 deg
    yi = cpsi==0    #index of Y coordinates (spsi = 1) psi = 90 deg


    Parameters
    ----------
    timestamps_2D
    scan_angle_definition

    Returns
    -------

    """

    # every 2D timestamp is duplicated to obtain the 1D timestamps
    try:
        timestamps_1D = np.sort(np.hstack((timestamps_2D, timestamps_2D)))
    except AttributeError:
        1/0
    n_1d = len(timestamps_1D)

    # compute cos(psi) and sin(psi) factors assuming orthogonal axes
    if scan_angle_definition == 'hipparcos':
        spsi = (np.arange(1, n_1d+1)+1)%2# % first Ra then Dec
        cpsi = (np.arange(1, n_1d+1)  )%2

        # indices of X and Y measurements
        xi = np.where(spsi==0)[0]    #index of X coordinates (cpsi = 1) psi =  0 deg
        yi = np.where(cpsi==0)[0]    #index of Y coordinates (spsi = 1) psi = 90 deg

    elif scan_angle_definition == 'gaia':
        cpsi = (np.arange(1, n_1d+1)+1)%2
        spsi = (np.arange(1, n_1d+1)  )%2

        # indices of X and Y measurements
        yi = np.where(spsi==0)[0]
        xi = np.where(cpsi==0)[0]

    return timestamps_1D, cpsi, spsi, xi, yi


def mass_from_semimajor_axis(a_m, p_day):
    """Return mass term in Kepler's law.

    M_0,1,2 = 4 pi^2 a_0,1,2^3 / P^2

    Parameters
    ----------
    a_m
    p_day

    Returns
    -------

    """

    mass_term = 4 * np.pi**2 * a_m**3/(p_day*day2sec)**2
    mass_kg = mass_term / Ggrav

    return mass_kg


def convert_from_linear_to_angular(a_m, absolute_parallax_mas):
    """Convert a linear quantity in meters to a angle in mas, given the absolute parallax.

    Parameters
    ----------
    a_m
    absolute_parallax_mas

    Returns
    -------

    """
    d_pc = 1./ (absolute_parallax_mas/1000.)
    a_rad = np.arctan2(a_m, d_pc*pc_m)
    a_mas = a_rad * rad2mas  # semimajor axis in mas
    return a_mas


def convert_from_angular_to_linear(a_mas, absolute_parallax_mas):
    """Convert a angle in mas to a linear quantity in meters, given the absolute parallax.

    Parameters
    ----------
    a_mas
    absolute_parallax_mas

    Returns
    -------

    """
    a_rad = a_mas/rad2mas
    d_pc = 1. / (absolute_parallax_mas / 1000.)
    a_m = np.tan(a_rad) * d_pc*pc_m
    # a_m = a_rad * d_pc*pc_m

    return a_m


def companion_mass_in_diluted_system(alpha_mas, absolute_parallax_mas, m1_kg, p_day, delta_mag,
                                     numeric_solution=True):
    """Return companion mass given photocenter orbit and delta_mag."""

    g_value = Ggrav / (4 * np.pi**2) * (p_day * day2sec)**2
    alpha_value = convert_from_angular_to_linear(alpha_mas, absolute_parallax_mas)
    beta_value = fractional_luminosity(0, delta_mag)

    if numeric_solution:
        alpha = alpha_value
        m1 = m1_kg
        beta = beta_value
        g = g_value

        zero_equation = lambda m2: g * (m1 + m2) - (alpha / (m2 / (m1 + m2) - beta)) ** 3  # == 0

        # scipyfmin minimizes the given function with a given starting value
        m2_kg = scipyfmin(zero_equation, m1, disp=False)

        return m2_kg

    else:
        alpha = alpha_value
        m1 = m1_kg
        beta = beta_value
        g = g_value

        m2_kg = np.array([-(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)/(3*(27*(alpha**3*m1**2 + beta**3*g*m1**3)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) + np.sqrt(-4*(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)**3 + (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2 + 2*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**2)/2 - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**(1./3)) - (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(3*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) - (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) + np.sqrt(-4*(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)**3 + (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2 + 2*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**2)/2 - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**(1./3)/3, -(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)/(3*((-1./2) - np.sqrt(3)*1j/2)*(27*(alpha**3*m1**2 + beta**3*g*m1**3)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) + np.sqrt(-4*(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)**3 + (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2 + 2*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**2)/2 - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**(1./3)) - (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(3*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) - ((-1./2) - np.sqrt(3)*1j/2)*(27*(alpha**3*m1**2 + beta**3*g*m1**3)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) + np.sqrt(-4*(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)**3 + (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2 + 2*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**2)/2 - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**(1./3)/3, -(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)/(3*((-1./2) + np.sqrt(3)*1j/2)*(27*(alpha**3*m1**2 + beta**3*g*m1**3)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) + np.sqrt(-4*(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)**3 + (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2 + 2*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**2)/2 - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**(1./3)) - (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(3*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) - ((-1./2) + np.sqrt(3)*1j/2)*(27*(alpha**3*m1**2 + beta**3*g*m1**3)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) + np.sqrt(-4*(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)**3 + (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2 + 2*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**2)/2 - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**(1./3)/3])
        # m2_kg = -(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)/(3*(27*(alpha**3*m1**2 + beta**3*g*m1**3)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) + np.sqrt(-4*(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)**3 + (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2 + 2*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**2)/2 - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**(1/3.)) - (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(3*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) - (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)) + np.sqrt(-4*(-3*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**2/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2)**3 + (27*(alpha**3*m1**2 + beta**3*g*m1**3)/(beta**3*g - 3*beta**2*g + 3*beta*g - g) - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2 + 2*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**2)/2 - 9*(2*alpha**3*m1 + 3*beta**3*g*m1**2 - 3*beta**2*g*m1**2)*(alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)/(2*(beta**3*g - 3*beta**2*g + 3*beta*g - g)**2) + (alpha**3 + 3*beta**3*g*m1 - 6*beta**2*g*m1 + 3*beta*g*m1)**3/(beta**3*g - 3*beta**2*g + 3*beta*g - g)**3)**(1/3.)/3

    if 0:
        # solve the equation using sympy

        alpha = sp.Symbol('alpha')
        beta = sp.Symbol('beta')
        g = sp.Symbol('g')
        m1 = sp.Symbol('m1')
        m2 = sp.Symbol('m2')
        zero_equation = g * (m1 + m2) - (alpha / (m2/(m1 + m2) - beta))**3 # == 0
        res = sp.solvers.solve(zero_equation, m2, check=False)
        print(sp.python(res))
        for i, sol in enumerate(res):
            print('Solution {}'.format(i))
            if i == 1:
                m2_kg = sol.evalf(subs={g: g_value, m1: m1_kg, beta: beta_value, alpha: alpha_value})
                return m2_kg

    return m2_kg


def pjGet_m2(m1_kg, a_m, P_day):
    """Return companion mass in kg.

    Parameters
    ----------
    m1_kg : float
        primary mass in kg
    a_m : float
        barycentric semimajor axis in meter
    P_day : float
        orbital period in days

    Returns
    -------

    """
    c = np.abs(4.*np.pi**2.*a_m**3./(P_day*day2sec)**2.)

    a = np.sqrt( c / Ggrav ) * m1_kg
    b = np.sqrt( c / Ggrav )

    m2_kg = (27.*a**2. + 3.*np.sqrt(3.)* np.sqrt(27.*a**4. +  4.*a**3.*b**3.) + 18.*a*b**3. + 2.*b**6.)**(1./3.) / (3.*2.**(1./3.)) - (2.**(1./3.)*(-6.*a*b - b**4.)) / (3.* (27.*a**2. + 3.*np.sqrt(3)*np.sqrt( 27.*a**4. + 4.*a**3.*b**3. ) + 18.*a*b**3. + 2.*b**6.)**(1./3.))+(b**2.)/3.

    if 0 == 1:
        # from sympy import Eq, Symbol, solve
        import sympy as sp
        # (a1_detection_mas/1.e3 * AU_m * d_pc)**3 * (4. * np.pi**2.) * (P_day*day2sec)**2. =  Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2.
        # m2_MJ = sp.Symbol('m2_MJ')
        # P_day = sp.Symbol('P_day')
        # a = (a1_detection_mas/1.e3 * AU_m * d_pc)**3 * (4. * np.pi**2.)
        # b = a * (P_day*day2sec)**2 / Ggrav
        # m2 = m2_MJ * MJ_kg
        # m1 = m1_MS*MS_kg
        a = sp.Symbol('a')
        p = sp.Symbol('p')
        G = sp.Symbol('G')
        m1 = sp.Symbol('m1')
        m2 = sp.Symbol('m2')
        # g1 = b -  (m2)**3 / ( m1 + m2 )**2
        # a_AU = a_m / AU_m #  in AU
        # a1_mas*d_pc*AU_m / 1e3 = a_m
        # p1 =     (4. * np.pi**2.)
        # p2 = (self.P_day*day2sec)**2
        # p = p2/p1*G
        # a_m = a1_detection_mas / 1.e3 * d_pc * AU_m
        # a = (a1_detection_mas / 1.e3 * d_pc * AU_m)**3

        # M/G =  m2**3 / ( m1 + m2 )**2
        # a = M * p
        # g1 = a - M*p
        g1 = p * m2**3  / ( m1 + m2 )**2 - a
        res = sp.solvers.solve( (g1), (m2))
        print(res)


    return m2_kg


def semimajor_axis_barycentre_angular(m1_MS, m2_MJ, P_day, plx_mas):
    """Return the semi-major axis, in milliarcseconds, of a primary object's orbit
    around the system barycenter.

    Parameters
    ----------
    m1_MS : `float`
        The mass of the primary, in solar masses.

    m2_MJ : `float`
        The mass of the secondary, in Jupiter masses.

    P_day : `float`
        The period of the secondary, in Earth days.

    plx_mas : `float`
        The parallax of the primary, in milliarcseconds.

    Returns
    ----------
    a_barycentre : `float`
        The apparent semi-major axis of the primary, in milliarcseconds.
    """
    # # mass term for the barycentric orbit of the primary mass
    # M = (Ggrav * (m2_MJ * MJ_kg)**3. / (m1_MS * MS_kg + m2_MJ * MJ_kg)**2.)
    #
    # # semimajor axis of the primary mass in meter
    # a_m = (M / (4. * np.pi**2.) * (P_day * day2sec)**2.)**(1./3.)

    a_m = semimajor_axis_barycentre_linear(m1_MS, m2_MJ, P_day)

    d_pc  = 1. / (plx_mas / 1000.)
    a_rad = np.arctan2(a_m, d_pc*pc_m)

    # semimajor axis in mas
    a_mas = a_rad * rad2mas

    return a_mas


def semimajor_axis_barycentre_linear(m1_MS, m2_MJ, P_day):
    """
    Get the semi-major axis, in meters, of a primary object's orbit around the
    system barycenter.

    Parameters
    ----------
    m1_MS : `float`
        The mass of the primary, in solar masses.

    m2_MJ : `float`
        The mass of the secondary, in Jupiter masses.

    P_day : `float`
        The period of the secondary, in Earth days.

    Returns
    ----------
    a_m_barycentre : `float`
        The physical semi-major axis of the primary, in meters.
    """
    M = (Ggrav * (m2_MJ * MJ_kg)**3.
         / (m1_MS * MS_kg + m2_MJ * MJ_kg)**2.) # mass term for the barycentric orbit of the primary mass
    a_m = (M / (4. * np.pi**2.) * (P_day * day2sec)**2.)**(1./3.)  # semimajor axis of the primary mass in m
    return a_m


def semimajor_axis_relative_angular(m1_MS, m2_MJ, P_day, plx_mas):
    """
    Get the semi-major axis, in milliarcseconds, of a secondary object's orbit
    around its primary.

    Parameters
    ----------
    m1_MS : `float`
        The mass of the primary, in solar masses.

    m2_MJ : `float`
        The mass of the secondary, in Jupiter masses.

    P_day : `float`
        The period of the secondary, in Earth days.

    plx_mas : `float`
        The parallax of the primary, in milliarcseconds.

    Returns
    ----------
    a_relative : `float`
        The apparent semi-major axis of the secondary, in milliarcseconds.
    """
    # a_rel_m = ((Ggrav * (m1_MS * MS_kg + m2_MJ * MJ_kg)
    #             / 4. / (np.pi**2.)
    #             * (P_day * day2sec)**2.)**(1./3.))
    #M = Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
    #a_m = ( M / (4. * np.pi**2.) * (P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
    a_rel_m = semimajor_axis_relative_linear(m1_MS, m2_MJ, P_day)
    d_pc  = 1./ (plx_mas / 1000.)
    a_rel_rad = np.arctan2(a_rel_m, d_pc * pc_m)
    a_rel_mas = a_rel_rad * rad2mas # semimajor axis in mas
    return a_rel_mas


def semimajor_axis_relative_linear(m1_MS, m2_MJ, P_day):
    """Get the semi-major axis, in meters, of a secondary object's orbit around
    its primary.

    Parameters
    ----------
    m1_MS : `float`
        The mass of the primary, in solar masses.

    m2_MJ : `float`
        The mass of the secondary, in Jupiter masses.

    P_day : `float`
        The period of the secondary, in Earth days.

    Returns
    ----------
    a_m_relative : `float`
        The physical semi-major axis of the secondary, in meters.
    """
    a_rel_m = ((Ggrav * (m1_MS * MS_kg + m2_MJ * MJ_kg)
                / 4. / (np.pi**2.)
                * (P_day * day2sec)**2.)**(1./3.))
    return a_rel_m


def secondary_mass_at_detection_limit( m1_MS, Period_day, d_pc, a1_detection_mas ):
    """
    formerly pjGet_DetectionLimits


    Parameters
    ----------
    m1_MS
    Period_day
    d_pc
    a1_detection_mas

    Returns
    -------

    """

    a_m = a1_detection_mas / 1.e3 * d_pc * AU_m
    m1_kg = m1_MS * MS_kg
    P_day = Period_day

    m2_kg = pjGet_m2( m1_kg, a_m, P_day )
    m2_MJ = m2_kg / MJ_kg
    return m2_MJ




def mean_anomaly(t_mjd, t_periastron_mjd, p_day):
    """Return mean anomaly at time t_mjd.

    Parameters
    ----------
    t_mjd : float
        time in MJD
    t_periastron_mjd : float
        Time of periastron passage in MJD
    p_day : float
        Orbital period in days

    Returns
    -------
    m_deg : float
        Mean anomaly

    """
    m_deg = np.rad2deg((t_mjd - t_periastron_mjd) * (2 * np.pi)/p_day)
    return m_deg


def eccentric_anomaly(ecc, t_mjd, t_periastron_mjd, p_day):
    """

    following MIKS-GA4FORS_v0.4/genetic/kepler-genetic.i

    Parameters
    ----------
    ecc
    t_mjd
    t_periastron_mjd
    p_day

    Returns
    -------

    """
    m_deg = mean_anomaly(t_mjd, t_periastron_mjd, p_day)
    M_rad = np.deg2rad(m_deg)
    if np.all(ecc) == 0.0:
        return M_rad
    else:
        E_rad = np.zeros(len(M_rad))

        E0_rad = M_rad + ecc*np.sin(M_rad)*(1+ecc*np.cos(M_rad)) #valeur initiale
        Enew_rad=E0_rad # initialissation a l'anomalie moyenne
        cnt=0  #compteur d'iterations
        E_rad_tmp = 1000.
        while (np.max(np.abs(Enew_rad-E_rad_tmp)) >1.e-8) & (cnt<200):
            E_rad_tmp = Enew_rad
            f   =  E_rad_tmp - ecc*np.sin(E_rad_tmp) - M_rad
            fp  =  1-ecc*np.cos(E_rad_tmp)#derivee de f par rapport a E
            fpp =  ecc*np.sin(E_rad_tmp)
            # //Enew_rad = E_rad_tmp - f/fp //
            # //Enew_rad = E_rad_tmp -2*fp/fpp - sqrt( (fp/fpp)^2 +f) bof
            Enew_rad = E_rad_tmp - 2*fp*f/(2*fp**2-f*fpp) #marche tres bien
            cnt += 1
        E_rad = E_rad_tmp
        return E_rad


def RadialVelocitiesConstants(k1_mps,om_rad,ecc):

    alpha_mps = +k1_mps*np.cos(om_rad)
    beta_mps  = -k1_mps*np.sin(om_rad)
    delta_mps = +k1_mps*ecc*np.cos(om_rad)

    return np.array([alpha_mps,beta_mps,delta_mps])


def TrueAnomaly(ecc, E_rad):
    # BUG FOUND 2016-02-08, NOT SURE WHERE THIS CAME FROM
    #     theta_rad_tmp = 2.*np.arctan( np.sqrt((1.+ecc)/(1.-ecc))*np.tan(E_rad/2.) )
    #     theta_rad = np.arctan2( np.cos(theta_rad_tmp), np.sin(theta_rad_tmp) )

    theta_rad = 2.*np.arctan( np.sqrt((1.+ecc)/(1.-ecc))*np.tan(E_rad/2.) )
    return theta_rad


def RadialVelocitiesKepler(alpha_mps,beta_mps,delta_mps,theta_rad):
    Vrad_mps   = alpha_mps * np.cos(theta_rad) + beta_mps * np.sin(theta_rad) + delta_mps
    return Vrad_mps


def EllipticalRectangularCoordinates(ecc, E_rad):
# /*
#  * DOCUMENT
#  *   EllipticalRectangularCoordinates(ecc,E_rad)
#  *
#  *   It computes the ellipses of the orbit for  \f$  i=0\f$  and \f$  \Omega=0\f$
#  *
#  *
#  *   - INPUT
#  *       - omega_rad Longitude of periastron expressed in radian
#  *       - ecc Eccentricity
#  *       - Tp_day Time of passage at periastron (julian date-2400000)
#  *       - P_day Period of the orbit
#  *       - t_day Date/time of the observations  (julian date-2400000)
#  *
#  *    OUTPUT
#  *       Position on the sky. Needs the Thieles-Innes coef
#  *
#  *
#  *
#  *  SEE ALSO EccentricAnomaly
#  */
  X = np.cos(E_rad) - ecc
  Y = np.sqrt(1.-ecc**2)*np.sin(E_rad)
  return np.array([X,Y])


def geometric_elements(thiele_innes_parameters):
    """Return geometrical orbit elements a, omega, OMEGA, i.

    Parameters
    ----------
    thiele_innes_constants : array or array of arrays
        Array of Thiele Innes constants [A,B,F,G] in milli-arcsecond

    Returns
    -------
    geometric_parameters : array
        Orbital elements [a_mas, omega_deg, OMEGA_deg, i_deg]

    """
    A = thiele_innes_parameters[0]
    B = thiele_innes_parameters[1]
    F = thiele_innes_parameters[2]
    G = thiele_innes_parameters[3]

    p = (A ** 2 + B ** 2 + G ** 2 + F ** 2) / 2.
    q = A * G - B * F

    a_mas = np.sqrt(p + np.sqrt(p ** 2 - q ** 2))
    # i_rad = math.acos(q/(a_mas**2.))
    # omega_rad = (math.atan2(B-F,A+G)+math.atan2(-B-F,A-G))/2.;
    # OMEGA_rad = (math.atan2(B-F,A+G)-math.atan2(-B-F,A-G))/2.;

    i_rad = np.arccos(q / (a_mas ** 2.))
    omega_rad = (np.arctan2(B - F, A + G) + np.arctan2(-B - F, A - G)) / 2.
    OMEGA_rad = (np.arctan2(B - F, A + G) - np.arctan2(-B - F, A - G)) / 2.

    i_deg = np.rad2deg(i_rad)
    omega_deg = np.rad2deg(omega_rad)
    OMEGA_deg = np.rad2deg(OMEGA_rad)
    # OMEGA_deg = np.rad2deg(np.unwrap(OMEGA_rad))

    if np.any(np.isnan(a_mas)):
        index = np.where(np.isnan(a_mas))[0]
        raise RuntimeError('nan detected: {} occurrences'.format(len(index)))

    # if isinstance(omega_deg, (list, tuple, np.ndarray)):
    #     index = np.where(omega_deg < 0.)[0]
    #     omega_deg[index] += 180.
    #
    # if isinstance(OMEGA_deg, (list, tuple, np.ndarray)):
    #     index = np.where(OMEGA_deg < 0.)[0]
    #     OMEGA_deg[index] += 180.

    geometric_parameters = np.array([a_mas, omega_deg, OMEGA_deg, i_deg])
    return geometric_parameters


def thiele_innes_constants(geometric_parameters):
    """Return A B F G in mas from the input of the geometrical elements

    Parameters
    ----------
    geometric_parameters : array
        [a_mas, omega_deg, OMEGA_deg, i_deg]

    Returns
    -------
    thiele_innes_parameters : array
        [A, B, F, G] in mas

    """
    a_mas     = geometric_parameters[0]
    omega_rad = np.deg2rad(geometric_parameters[1])
    OMEGA_rad = np.deg2rad(geometric_parameters[2])
    i_rad     = np.deg2rad(geometric_parameters[3])

    A = a_mas * (np.cos(OMEGA_rad)*np.cos(omega_rad)  - np.sin(OMEGA_rad)*np.sin(omega_rad)*np.cos(i_rad))
    B = a_mas * (np.sin(OMEGA_rad)*np.cos(omega_rad)  + np.cos(OMEGA_rad)*np.sin(omega_rad)*np.cos(i_rad))
    F = a_mas * (-np.cos(OMEGA_rad)*np.sin(omega_rad) - np.sin(OMEGA_rad)*np.cos(omega_rad)*np.cos(i_rad))
    G = a_mas * (-np.sin(OMEGA_rad)*np.sin(omega_rad) + np.cos(OMEGA_rad)*np.cos(omega_rad)*np.cos(i_rad))

    thiele_innes_parameters = np.array([A, B, F, G])
    return thiele_innes_parameters


def astrom_signal(t_day, psi_deg, ecc, P_day, Tp_day, TIC):
    #USAGE of pseudo eccentricity
    # a = [pecc,P_day,Tp_day,A,B,F,G]
    # input: xp = structure containing dates and baseline orientations of measurements
    #         a = structure containing aric orbit parameters
    # output: phi = displacment angle in mas
    # pecc = a(1)    #ecc = abs(double(atan(pecc)*2/pi))    # ecc = retrEcc( pecc )

    # psi_rad = psi_deg *2*np.pi/360
    psi_rad = np.deg2rad(psi_deg)


    # compute eccentric anomaly
    E_rad = eccentric_anomaly(ecc, t_day, Tp_day, P_day)

    # compute orbit projected on the sky
    if np.all(ecc == 0):
        X = np.cos(E_rad)
        Y = np.sin(E_rad)
    else:
        X = np.cos(E_rad)-ecc
        Y = np.sqrt(1.-ecc**2)*np.sin(E_rad)
    #compute phi
    # A = TIC[0]
    # B = TIC[1]
    # F = TIC[2]
    # G = TIC[3]
    # phi = (A*np.sin(psi_rad)+B*np.cos(psi_rad))*X + (F*np.sin(psi_rad)+G*np.cos(psi_rad))*Y
    phi = (TIC[0]*np.sin(psi_rad)+TIC[1]*np.cos(psi_rad))*X + (TIC[2]*np.sin(psi_rad)+TIC[3]*np.cos(psi_rad))*Y

    # return np.array(phi)
    return phi


def astrom_signalFast(t_day, spsi, cpsi, ecc, P_day, T0_day, TIC, scan_angle_definition='hipparcos'):
    """Return astrometric orbit signal.

    Parameters
    ----------
    t_day
    spsi
    cpsi
    ecc
    P_day
    T0_day
    TIC

    Returns
    -------
    phi : numpy array
        Orbit signal along scan angle psi.

    """

    # compute eccentric anomaly
    E_rad = eccentric_anomaly(ecc, t_day, T0_day, P_day)

    # compute orbit projected on the sky
    if np.all(ecc == 0):
        X = np.cos(E_rad)
        Y = np.sin(E_rad)
    else:
        X = np.cos(E_rad)-ecc
        Y = np.sqrt(1.-ecc**2)*np.sin(E_rad)

    # see Equation 8 in Sahlmann+2011
    if scan_angle_definition == 'hipparcos':
        phi = (TIC[0]*spsi + TIC[1]*cpsi)*X + (TIC[2]*spsi + TIC[3]*cpsi)*Y
    elif scan_angle_definition == 'gaia':
        #         A            B                   F           G
        phi = (TIC[0]*cpsi + TIC[1]*spsi)*X + (TIC[2]*cpsi + TIC[3]*spsi)*Y

    return phi


def get_ephemeris(center='g@399', target='0', start_time=None, stop_time=None, step_size='5d',
                  verbose=True, out_dir=None, vector_table_output_type=1, output_units='AU-D',
                  overwrite=False, reference_plane='FRAME'):
    """Query the JPL Horizons web interface to return the X,Y,Z position of the target body
    relative to the center body.


    Parameters
    ----------
    center : str
        Horizons object identifier, default is Earth Center 'g@399'
    target : str
        Horizons object identifier, default is Solar System Barycenter '0'
    start_time : astropy time instance
    stop_time : astropy time instance
    step_size : string, default is '1d' for 1 day steps
    verbose : bool
    out_dir : str
    vector_table_output_type
    output_units
    overwrite
    reference_plane : str
        reference_plane = 'FRAME' is for Earth mean equator and equinox

    Returns
    -------
    xyzdata : astropy table


    References
    ----------
    See Horizons_doc.pdf available at https://ssd.jpl.nasa.gov/?horizons#email
    Documentation can also be obtained by sending en email with subject "BATCH-LONG" to
    horizons@ssd.jpl.nasa.gov

    """
    global ephemeris_dir

    if start_time is None:
        start_time = Time(1950.0, format='jyear')
    if stop_time is None:
        stop_time = Time(2025.0, format='jyear')

    if out_dir is not None:
        ephemeris_dir = out_dir

    if output_units not in ['AU-D', 'KM-S', 'KM-D']:
        raise NotImplementedError()

    if reference_plane not in ['ECLIPTIC', 'FRAME', 'B']: # last is BODY EQUATOR
        raise NotImplementedError()

    if vector_table_output_type not in np.arange(6)+1:
        raise NotImplementedError()

    horizons_file_seed = '{}_{}_{}_{}_{}'.format(center, target, start_time, stop_time, step_size)
    out_file = os.path.join(ephemeris_dir, horizons_file_seed + '.txt')

    if verbose:
        print('Getting ephemeris {}'.format(horizons_file_seed))

    if (not os.path.isfile(out_file)) or overwrite:
        # run Horizons query
        url = "https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=l&TABLE_TYPE='VECTORS'&CSV_FORMAT='YES'"
        url += "&CENTER='{}'".format(center)
        url += "&COMMAND='{}'".format(target)
        url += "&START_TIME='{}'".format(start_time.isot.split('T')[0])
        url += "&STOP_TIME='{}'".format(stop_time.isot.split('T')[0])
        url += "&STEP_SIZE='{}'".format(step_size)
        url += "&SKIP_DAYLT='NO'"
        url += "&OUT_UNITS='{}'".format(output_units)
        url += "&VEC_TABLE='{}'".format(vector_table_output_type)
        url += "&REF_PLANE='{}'".format(reference_plane)

        if verbose:
            print(url)
        try:
            url_stream = urlopen(url)
        except HTTPError as e:
            print("Unable to open URL:", e)
            sys.exit(1)

        content = url_stream.read()
        url_stream.close()

        with open(out_file, 'wb') as ephemeris:
            ephemeris.write(content)

    xyzdata = read_ephemeris(horizons_file_seed, overwrite=overwrite, ephemeris_path=ephemeris_dir)
    return xyzdata



def read_ephemeris(horizons_file_seed, overwrite=False, ephemeris_path=None, verbose=False):
    """
    Read ephemeris file obtained from the JPL HORIZONS system

    TODO: clean up computation of data_start and data_end

    :param horizons_file_seed:
    :return:
    """

    if ephemeris_path is None:
        ephemeris_path = ephemeris_dir



    fits_file = os.path.join(ephemeris_path, horizons_file_seed + '_XYZ.fits')
    if (not os.path.isfile(fits_file)) or overwrite:
        eph_file = os.path.join(ephemeris_path, horizons_file_seed + '.txt')
        f_rd = open(eph_file, 'r')
        # file_lines = f_rd.readlines()[0].split('\r')
        file_lines = f_rd.readlines()
        f_rd.close()
        # for i in range(len(file_lines)):
        #     line = file_lines[i]
        #     print('{} {}'.format(i, line))
        #     if line.strip()=='':
        #         print('{} Empty line detected'.format(i))

        index_start = [i for i in range(len(file_lines)) if "$$SOE" in file_lines[i]][0]
        index_end   = [i for i in range(len(file_lines)) if "$$EOE" in file_lines[i]][0]
        # n_blank_lines = len([i for i in range(index_start) if (file_lines[i] == '' or file_lines[i] == ' ' or file_lines[i].strip() == '\n')])
        n_blank_lines = len([i for i in range(index_start) if (file_lines[i].strip() in ['\n',''])])
        # data_start = index_start + 1
        data_start = index_start - n_blank_lines + 1
        data_end = data_start + index_end - index_start -1
        # data_end = index_end - 1
        header_start = index_start - n_blank_lines -2
        if verbose:
            print('Number of blank lines found before data: {}'.format(n_blank_lines))
            print('index_start: {}'.format(index_start))
            print('index_end: {}'.format(index_end))
            print('data_start: {}'.format(data_start))
            print('data_end: {}'.format(data_end))
            print('header start: {}'.format(header_start))
        xyzdata = Table.read(eph_file, format='ascii.basic', delimiter=',', data_start = data_start,
                             data_end=data_end, guess=False, comment='mycomment95', header_start = header_start)
        xyzdata.write(fits_file, format = 'fits', overwrite=True)
        # xyzdata = Table.read(eph_file, format='ascii.no_header', delimiter=',', data_start = data_start,
        #                      data_end=data_end, names=('JD','ISO','X','Y','Z','tmp'), guess=False, comment='mycomment95')
        # xyzdata['JD','X','Y','Z'].write(fits_file, format = 'fits')
    else:
        xyzdata = Table.read(fits_file, format = 'fits')

    for colname in xyzdata.colnames:
        if 'col' in colname:
            xyzdata.remove_column(colname)

    # xyzdata.rename_column('JDTDB', 'JD')

    return xyzdata


def get_parallax_factors(ra_deg, dec_deg, time_jd, horizons_file_seed=None, verbose=False,
                         instrument=None, overwrite=False):
    """

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    time_jd : ndarray
        Array of times in Julian Day format
    horizons_file_seed : str
        Optional input of pre-existing ephemeris file from JPL Horizons
    verbose : bool
        verbosity
    instrument : str
        Optional argument when using pre-existing ephemeris file
    overwrite : bool
        Whether to overwrite existing products

    Returns
    -------
        [parallax_factor_ra, parallax_factor_dec] : ndarray
        Arrays holding the parallax factors
    """

    ephFactor = -1
    ra_rad = np.deg2rad(ra_deg)
    de_rad = np.deg2rad(dec_deg)

    if instrument is not None:
        instr = np.unique(instrument)
        Nepoch = len(instrument)
        Xip_val = np.zeros(Nepoch)
        Yip_val = np.zeros(Nepoch)
        Zip_val = np.zeros(Nepoch)

        for ins in instr:
            idx = np.where( instrument == ins )[0]
            if verbose:
                print('Getting Parallax factors for %s using Seed: \t%s' % (ins, DEFAULT_EPHEMERIS_DICTIONARY[ins]))
            xyzdata = read_ephemeris(DEFAULT_EPHEMERIS_DICTIONARY[ins])
            Xip = interp1d(xyzdata['JD'],xyzdata['X'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
            Yip = interp1d(xyzdata['JD'],xyzdata['Y'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
            Zip = interp1d(xyzdata['JD'],xyzdata['Z'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
            try:
                Xip_val[idx] = Xip(time_jd[idx])
                Yip_val[idx] = Yip(time_jd[idx])
                Zip_val[idx] = Zip(time_jd[idx])
            except ValueError:
                print('Error in time interpolation for parallax factors: range %3.1f--%3.2f (%s--%s)\n' % (np.min(time_jd[idx]), np.max(time_jd[idx]), Time(np.min(time_jd[idx]), format='jd', scale='utc').iso, Time(np.max(time_jd[idx]), format='jd', scale='utc').iso)),
                print('Ephemeris file contains data from %s to %s' % (Time(np.min(xyzdata['JD']),format='jd').iso, Time(np.max(xyzdata['JD']),format='jd').iso))
                pdb.set_trace()
                1/0

            parallax_factor_ra  = ephFactor* ( Xip_val*np.sin(ra_rad) - Yip_val*np.cos(ra_rad) )
            parallax_factor_dec =  ephFactor*(( Xip_val*np.cos(ra_rad) + Yip_val*np.sin(ra_rad) )*np.sin(de_rad) - Zip_val*np.cos(de_rad))

    else:
        if horizons_file_seed is None:
            xyzdata = get_ephemeris(verbose=verbose, overwrite=overwrite)

        # if verbose:
        #     print('Getting Parallax factors using Seed: \t%s' % horizons_file_seed)
        else:
            xyzdata = read_ephemeris(horizons_file_seed)
        Xip = interp1d(xyzdata['JDTDB'],xyzdata['X'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
        Yip = interp1d(xyzdata['JDTDB'],xyzdata['Y'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
        Zip = interp1d(xyzdata['JDTDB'],xyzdata['Z'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)

        try:
            parallax_factor_ra  = ephFactor* (Xip(time_jd) * np.sin(ra_rad) - Yip(time_jd) * np.cos(ra_rad))
            parallax_factor_dec =  ephFactor*((Xip(time_jd) * np.cos(ra_rad) + Yip(time_jd) * np.sin(ra_rad)) * np.sin(de_rad) - Zip(time_jd) * np.cos(de_rad))
        except ValueError:
            raise ValueError('Error in time interpolation for parallax factors: \n'
                             'requested range {:3.1f}--{:3.1f} ({}--{})\n'
                             'available range {:3.1f}--{:3.1f} ({}--{})'.format(np.min(time_jd), np.max(time_jd), Time(np.min(time_jd), format='jd', scale='utc').iso, Time(np.max(time_jd), format='jd', scale='utc').iso, np.min(xyzdata['JDTDB']), np.max(xyzdata['JDTDB']), Time(np.min(xyzdata['JDTDB']), format='jd', scale='utc').iso, Time(np.max(xyzdata['JDTDB']), format='jd', scale='utc').iso
                                                                                ) )
    return [parallax_factor_ra, parallax_factor_dec]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def pjGetOrbitFast(P_day=100, ecc=0, m1_MS=1, m2_MJ = 1, omega_deg=0, OMEGA_deg=0, i_deg=45, T0_day = 0, plx_mas = 25, t_MJD='', spsi='', cpsi='', verbose=0):
# /* DOCUMENT ARV -- simulate fast 1D astrometry for planet detection limits
#    written: J. Sahlmann   18 May 2015   ESAC
# */
    m2_MS = m2_MJ * MJ2MS
    d_pc  = 1./ (plx_mas/1000.)
    #**************ASTROMETRY********************************************************
    M = Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
    a_m = ( M / (4. * np.pi**2.) * (P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
    a_rad = np.arctan2(a_m,d_pc*pc_m)
    a_mas = a_rad * rad2mas # semimajor axis in mas
    TIC     = thiele_innes_constants([a_mas   , omega_deg     , OMEGA_deg, i_deg]) #Thiele-Innes constants
    phi1 = astrom_signalFast(t_MJD,spsi,cpsi,ecc,P_day,T0_day,TIC)
    return phi1


def dcr_coefficients(aux):
    """Return DCR parameters following Sahlmann+13.

    Parameters
    ----------
    aux : astropy table
        Table containing columns with predefined names.

    Returns
    -------

    """
    temp = aux['temperature'].data  # Celsius
    pres = aux['pressure'].data  # mbar

    f3m = (1. - (temp - 11.) / (273. + 11.)) * (1. + (pres - 744.) / 744.)

    # zenith angle
    z_rad = np.deg2rad(90. - aux['tel_altitude'].data)

    lat_rad = np.deg2rad(aux['geo_latitude'].data)
    dec_rad = np.deg2rad(aux['dec'].data)
    azi_rad = np.deg2rad(aux['tel_azimuth'].data)

    # hour angle
    ha_rad = [sla.slalib.sla_pda2h(lat_rad[i], dec_rad[i], azi_rad[i])[0] for i in range(len(dec_rad))]

    # parallactic angle
    pa_rad = [sla.slalib.sla_pa(ha_rad[i], dec_rad[i], lat_rad[i]) for i in range(len(dec_rad))]

    f1xm = f3m * np.tan(z_rad) * np.sin(pa_rad)
    f1ym = f3m * np.tan(z_rad) * np.cos(pa_rad)

    #     % DCR parameter 1
    xfactor = 1
    yfactor = 1
    xDCRfactor = np.array(xfactor * np.mat(f1xm).T).flatten()
    yDCRfactor = np.array(yfactor * np.mat(f1ym).T).flatten()

    return xDCRfactor, yDCRfactor


class ImagingAstrometryData(object):
    """Structure class for 2D imaging astrometry."""



    def __init__(self, data_table, out_dir=None, data_type='2d', time_column_name='MJD',
                 simbad_object_name=None):
        """

        Parameters
        ----------
        data_table
        out_dir
        data_type
        """

        required_data_table_columns = [time_column_name, 'frame', 'OB']
        for column_name in required_data_table_columns:
            if column_name not in data_table.colnames:
                raise ValueError('Input table has to have a column named: {}'.format(column_name))

        # sort data table by increasing time
        self.time_column_name = time_column_name
        self.simbad_object_name = simbad_object_name
        self.data_type = data_type
        self.scan_angle_definition = 'hipparcos'

        data_table.sort(self.time_column_name)

        self.data_table = data_table
        # self.epoch_data = data_table
        self.number_of_frames = len(np.unique(self.data_table['frame']))
        self.number_of_observing_blocks = len(np.unique(self.data_table['OB']))
        self.observing_time_span_day = np.ptp(data_table[self.time_column_name])

        if data_type=='2d':
            # unique Julian dates of observations, i.e. of 2D astrometry
            self.observing_times_2D_MJD, unique_index = np.unique(np.array(data_table[self.time_column_name]), return_index=True)
            self.data_2D = self.data_table[unique_index]
            self.number_of_1D_measurements = 2 * len(self.data_2D)
        else:
            self.data_1D = self.data_table
            self.number_of_1D_measurements = len(self.data_1D)


        if out_dir is not None:
            self.out_dir = out_dir
        else:
            self.out_dir = os.getcwd()

    def __str__(self):
        """Return string describing the instance."""
        description = '\nNumber of OBs: \t {}'.format(self.number_of_observing_blocks)
        description += '\nNumber of frames / measurements: \t {} / {}'.format(self.number_of_frames,
                                                                              self.number_of_1D_measurements)
        description += '\nObservation time span: \t {:3.1f} days'.format(self.observing_time_span_day)
        return description

    def set_object_coordinates(self, RA_deg=None, Dec_deg=None, overwrite=False):
        if (self.simbad_object_name is None) & (RA_deg is None) & (Dec_deg is None):
            print('Error: provide simbad name or coordinates')
            1/0
        elif (RA_deg is not None) & (Dec_deg is not None):
            self.RA_deg = RA_deg
            self.Dec_deg = Dec_deg
            return

        elif self.simbad_object_name is not None:

            object_string = self.simbad_object_name.replace(' ','')
            outFile = os.path.join(self.out_dir,'%s_simbad_parameters.txt' % object_string)
        if (not(os.path.isfile(outFile))) | (overwrite is True):
            mySimbad = Simbad()
            mySimbad.add_votable_fields('ra(d)','dec(d)','pmdec','pmra','parallax','sptype')
            pt = mySimbad.query_object(self.simbad_object_name)
            pt.write(outFile, format='ascii.basic',delimiter=',')
        else:
            pt = Table.read(outFile,format='ascii.basic',delimiter=',')

        self.simbad_object_parameters = pt
        self.RA_deg  = np.float(self.simbad_object_parameters['RA_d'])
        self.Dec_deg = np.float(self.simbad_object_parameters['DEC_d'])

        #         for c in ['RA_d','DEC_d','PMDEC','PMRA','PLX_VALUE','SP_TYPE']:

    def set_five_parameter_coefficients(self, earth_ephemeris_file_seed=None, verbose=False, reference_epoch_MJD=None, overwrite=False):
        """Set the coefficients of the five linear parameters, i.e. parallax factors and 0,1's for
        coordinates.

        Parameters
        ----------
        earth_ephemeris_file_seed
        verbose
        reference_epoch_MJD
        overwrite

        Returns
        -------

        """
        required_attributes = ['RA_deg', 'Dec_deg']
        for attribute_name in required_attributes:
            if hasattr(self, attribute_name) is False:
                raise ValueError('Instance has to have a attribute named: {}'.format(attribute_name))


        # TODO
        # clarify use of tdb here!
        observing_times_2D_TDB_JD = Time(self.observing_times_2D_MJD, format='mjd', scale='utc').tdb.jd

        # compute parallax factors, this is a 2xN_obs array
        observing_parallax_factors = get_parallax_factors(self.RA_deg, self.Dec_deg, observing_times_2D_TDB_JD, horizons_file_seed=earth_ephemeris_file_seed, verbose=verbose, overwrite=overwrite)

        # set reference epoch for position and computation of proper motion coefficients tspsi and tcpsi
        if reference_epoch_MJD is None:
            self.reference_epoch_MJD = np.mean(self.observing_times_2D_MJD)
        else:
            self.reference_epoch_MJD = reference_epoch_MJD

        # time relative to reference epoch in years for proper motion coefficients
        observing_relative_time_2D_year = (self.observing_times_2D_MJD - self.reference_epoch_MJD)/year2day

        observing_relative_time_1D_year, observing_1D_cpsi, observing_1D_spsi, self.observing_1D_xi, self.observing_1D_yi = get_cpsi_spsi_for_2Dastrometry(observing_relative_time_2D_year)

        observing_1D_tcpsi = observing_1D_cpsi * observing_relative_time_1D_year
        observing_1D_tspsi = observing_1D_spsi * observing_relative_time_1D_year

        observing_1D_ppfact = np.zeros(self.number_of_1D_measurements)
        observing_1D_ppfact[self.observing_1D_xi] = observing_parallax_factors[0]
        observing_1D_ppfact[self.observing_1D_yi] = observing_parallax_factors[1]

        self.five_parameter_coefficients_table = Table(np.array([observing_1D_cpsi,observing_1D_spsi,observing_1D_ppfact,observing_1D_tcpsi,observing_1D_tspsi]).T, names=('cpsi','spsi','ppfact','tcpsi','tspsi'))
        self.five_parameter_coefficients_array = np.array([self.five_parameter_coefficients_table[c].data for c in self.five_parameter_coefficients_table.colnames])
        self.observing_relative_time_1D_year = observing_relative_time_1D_year



    def set_linear_parameter_coefficients(self, earth_ephemeris_file_seed=None, verbose=False, reference_epoch_MJD=None):

        if not hasattr(self, 'five_parameter_coefficients'):
            self.set_five_parameter_coefficients(earth_ephemeris_file_seed=earth_ephemeris_file_seed, verbose=verbose, reference_epoch_MJD=reference_epoch_MJD)


        if ('fx[1]' in self.data_2D.colnames) & ('fx[2]' in self.data_2D.colnames):
            # the VLT/FORS2 case with a DCR corrector
            tmp_2D = self.data_2D[self.time_column_name,'fx[1]','fy[1]','fx[2]','fy[2]'] #,'RA*_mas','DE_mas','sRA*_mas','sDE_mas','OB','frame']
        elif ('fx[1]' in self.data_2D.colnames) & ('fx[2]' not in self.data_2D.colnames):
            # for GTC/OSIRIS, Gemini/GMOS-N/GMOS-S, VLT/HAWK-I
            tmp_2D = self.data_2D[self.time_column_name, 'fx[1]', 'fy[1]']
        elif ('fx[1]' not in self.data_2D.colnames) & ('fx[2]' not in self.data_2D.colnames):
            # anything else, e.g. RECONS, there is no DCR correction to be applied
            # tmp_2D = self.data_2D[[self.time_column_name]]
            self.linear_parameter_coefficients_table = self.five_parameter_coefficients_table
            self.linear_parameter_coefficients_array = np.array(
                [self.linear_parameter_coefficients_table[c].data for c in
                 self.linear_parameter_coefficients_table.colnames])
            return

        tmp_1D = tablevstack( (tmp_2D,tmp_2D) )
        tmp_1D.sort(self.time_column_name)


        # sign factors to get DCR coefficients right
        xfactor = -1
        yfactor =  1


        if 'fx[1]' in self.data_2D.colnames:
            tmp_1D.add_column(Column(name='rho_factor',data=np.zeros(len(tmp_1D))))
            tmp_1D['rho_factor'][self.observing_1D_xi] = xfactor * tmp_1D['fx[1]'][self.observing_1D_xi]
            tmp_1D['rho_factor'][self.observing_1D_yi] = yfactor * tmp_1D['fy[1]'][self.observing_1D_yi]
        if 'fx[2]' in self.data_2D.colnames:
            tmp_1D.add_column(Column(name='d_factor',data=np.zeros(len(tmp_1D))))
            tmp_1D['d_factor'][self.observing_1D_xi] = xfactor * tmp_1D['fx[2]'][self.observing_1D_xi]
            tmp_1D['d_factor'][self.observing_1D_yi] = yfactor * tmp_1D['fy[2]'][self.observing_1D_yi]

        if self.instrument == 'FORS2':
            self.dcr_parameter_coefficients_table = tmp_1D['rho_factor','d_factor']
        else:
            self.dcr_parameter_coefficients_table = tmp_1D[['rho_factor']]

        self.dcr_parameter_coefficients_array = np.array([self.dcr_parameter_coefficients_table[c].data for c in self.dcr_parameter_coefficients_table.colnames])


        self.linear_parameter_coefficients_table = tablehstack((self.five_parameter_coefficients_table, self.dcr_parameter_coefficients_table))
        self.linear_parameter_coefficients_table = tablehstack((self.five_parameter_coefficients_table, self.dcr_parameter_coefficients_table))
        self.linear_parameter_coefficients_array = np.array([self.linear_parameter_coefficients_table[c].data for c in self.linear_parameter_coefficients_table.colnames])

    def set_data_1D(self, earth_ephemeris_file_seed=None, verbose=False, reference_epoch_MJD=None):
        tmp_2D = self.data_2D[self.time_column_name,'RA*_mas','DE_mas','sRA*_mas','sDE_mas','OB','frame']
        tmp_1D = tablevstack( (tmp_2D,tmp_2D) )
        tmp_1D.sort(self.time_column_name)


        if not hasattr(self, 'linear_parameter_coefficients'):
            self.set_linear_parameter_coefficients(earth_ephemeris_file_seed=earth_ephemeris_file_seed, verbose=verbose, reference_epoch_MJD=reference_epoch_MJD)

        data_1D = tmp_1D[[self.time_column_name]]
        # astrometric measurement ('abscissa') and uncertainty
        data_1D.add_column(Column(name='da_mas',data=np.zeros(len(data_1D))))
        data_1D.add_column(Column(name='sigma_da_mas',data=np.zeros(len(data_1D))))

        data_1D['da_mas'][self.observing_1D_xi] = tmp_1D['RA*_mas'][self.observing_1D_xi]
        data_1D['da_mas'][self.observing_1D_yi] = tmp_1D['DE_mas'][self.observing_1D_yi]
        data_1D['sigma_da_mas'][self.observing_1D_xi] = tmp_1D['sRA*_mas'][self.observing_1D_xi]
        data_1D['sigma_da_mas'][self.observing_1D_yi] = tmp_1D['sDE_mas'][self.observing_1D_yi]

        for col in ['OB','frame']:
            data_1D[col] = tmp_1D[col]

        linear_parameter_coefficients_table = self.linear_parameter_coefficients_table
#         linear_parameter_coefficients.remove_column(self.time_column_name)

        self.data_1D = tablehstack((data_1D, linear_parameter_coefficients_table))
        self.observing_times_1D_MJD = self.data_1D[self.time_column_name].data #np.array(data_table[self.time_column_name])


def get_theta_best_genome(best_genome_file, reference_time_MJD, theta_names, m1_MS, instrument=None,
                          verbose=False):
    """

    :param best_genome_file:
    :param reference_time_MJD:
    :param theta_names:
    :param m1_MS:
    :param instrument:
    :param verbose:
    :return:
    """
    parameters = []

    best_genome = Table.read(best_genome_file, format='ascii.basic', data_start=2, delimiter=',', guess=False)

    if instrument.lower() != 'fors2':
        best_genome.remove_column('d_mas')

    # if verbose:
    if 0:
        for i in range(len(best_genome)):
            for c in best_genome.colnames:
                print('Planet %d: %s \t %3.3f' % (i+1, c, best_genome[c][i]))

    thiele_innes_constants = np.array([best_genome[c] for c in ['A','B','F','G']])

    a_mas, omega_deg, OMEGA_deg, i_deg = geometric_elements(thiele_innes_constants)
    d_pc  = 1./ (best_genome['plx_mas'].data.data /1000.)
    P_day = best_genome['P_day'].data.data
    a_m = a_mas / 1.e3 * d_pc * AU_m
    m1_kg = m1_MS * MS_kg
    m2_kg = pjGet_m2( m1_kg, a_m, P_day )
    # m2_kg = keplerian_secondary_mass( m1_kg, a_m, P_day )
    m2_MS = m2_kg / MS_kg
    # print(m2_MS)
    m2_MJ = m2_kg / MJ_kg
    TRef_MJD = reference_time_MJD

    # MIKS-GA computes T0 relative to the average time
    if verbose:
        for i in range(len(best_genome)):
            print('Planet %d: Phi0 = %f' % (i+1,best_genome['Tp_day'][i]))
            print('Planet %d: m2_MJ = %f' % (i+1, m2_MJ[i]))

    best_genome['Tp_day'] += TRef_MJD

    best_genome['a_mas'] = a_mas
    best_genome['omega_deg'] = omega_deg
    best_genome['i_deg'] = i_deg
    best_genome['OMEGA_deg'] = OMEGA_deg
    best_genome['m1_MS'] = m1_MS
    best_genome['m2_MS'] = m2_MS

    # col_list = theta_names #np.array(['P_day','ecc','m1_MS','m2_MS','omega_deg','Tp_day','dRA0_mas','dDE0_mas','plx_mas','muRA_mas','muDE_mas','rho_mas','d_mas','OMEGA_deg','i_deg'])

    for i in range(len(best_genome)):
        # generate dictionary
        theta = {c: best_genome[c][i] for c in best_genome.colnames}
        parameters.append(theta)

    if verbose:
        for i in range(len(best_genome)):
            theta = parameters[i]
            for key,value in theta.items():
                print('Planet %d: Adopted: %s \t %3.3f' % (i, key, value))

    # return theta_best_genome
    return parameters
