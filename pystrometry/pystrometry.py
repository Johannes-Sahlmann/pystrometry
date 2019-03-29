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
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import pylab as pl
from astropy import constants as const
from astropy.table import Table, Column
import astropy.units as u
from scipy.interpolate import *
import pdb
from matplotlib.lines import Line2D
from astropy.time import Time
from astropy.table import vstack as tablevstack
from astropy.table import hstack as tablehstack
from astroquery.simbad import Simbad
try:
    import pyslalib as sla
except ImportError:
    pass
import sys
if sys.version_info[0] == 3:
    # import urllib.request as urllib
    from urllib.request import urlopen
    from urllib.error import HTTPError
import pickle

from linearfit import linearfit
# from modules.functionsAndClasses import *
# from .functionsAndClasses import *
# import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../'))
# from astrohelpers.functions_and_classes import *


# #***********CONSTANTS*********************************************************
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

ephDict = {'Spitzer': 'horizons_XYZ_2003-2020_EQUATORIAL_Spitzer_1day_csv', 
'HST': 'horizons_XYZ_1990-2016_EQUATORIAL_HST_1day_csv', 
'WISE': 'horizons_XYZ_2009-2016_EQUATORIAL_WISE_1day_csv',
'JWST': 'horizons_XYZ_2012-2023_EQUATORIAL_JWST_1day_csv',
'L2': 'horizons_XYZ_1990-2035_EQUATORIAL_L2_1day_csv',
'Earth': 'horizons_XYZ_1990-2035_EQUATORIAL_Eart1day_csv'}

global ephemeris_dir
ephemeris_dir = '/Users/jsahlmann/astro/palta/processing/data/earthEphemeris/'


def fractional_luminosity( mag1 , mag2 ):
    """
    defining fraction luminosity of masses M1 and M2 as beta = L2/(L1+L2) and
    mag1-mag2=-2.5 log10(L1/L2), we find
    beta = 1/(1+10^(mag2-mag1))

    :param mag1:
    :param mag2:
    :return:
    """
    return 1./(1. + 10.**(0.4*(mag2-mag1)))


def fractional_mass(m1, m2):
    """
    computes fractional mass
    getB(m1,m2) returns m2/(m1+m2)

    :param m1:
    :param m2:
    :return:
    """
    return m2/(m1+m2)


class OrbitSystem(object):
    "2014-01-29  JSA ESAC"
    def __init__(self, P_day=100, ecc=0, m1_MS=1, m2_MJ=1, omega_deg=0., OMEGA_deg=0., i_deg=90., Tp_day=0., RA_deg=0., DE_deg=0., plx_mas=25., muRA_mas=20., muDE_mas=50., gamma_ms=0., rvLinearDrift_mspyr=None, rvQuadraticDrift_mspyr=None, rvCubicDrift_mspyr=None, Tref_MJD=None,
                 attribute_dict=None):
        if attribute_dict is not None:
            for key, value in attribute_dict.items():
                setattr(self, key, value)
                print('Setting {} to {}'.format(key, value))

                # set defaults
            default_dict = {'gamma_ms': 0.,
                            'rvLinearDrift_mspyr': 0.,
                            'rvQuadraticDrift_mspyr': 0,
                            'rvCubicDrift_mspyr': 0,
                            'scan_angle_definition': 'hipparcos'
                            }
            for key, value in default_dict.items():
                if key not in attribute_dict.keys():
                    setattr(self, key, value)

        else:
            self.P_day = P_day
            self.ecc = ecc
            self.m1_MS= m1_MS
            self.m2_MJ= m2_MJ
            self.omega_deg = omega_deg
            self.OMEGA_deg = OMEGA_deg
            self.i_deg = i_deg
            self.Tp_day = Tp_day
            self.RA_deg = RA_deg
            self.DE_deg = DE_deg
            self.plx_mas = plx_mas
            self.muRA_mas = muRA_mas
            self.muDE_mas = muDE_mas
            self.gamma_ms = gamma_ms
            self.rvLinearDrift_mspyr = rvLinearDrift_mspyr
            self.rvQuadraticDrift_mspyr = rvQuadraticDrift_mspyr
            self.rvCubicDrift_mspyr = rvCubicDrift_mspyr
            self.Tref_MJD = Tref_MJD

        self.m2_MS = self.m2_MJ * MJ_kg/MS_kg




    def pjGetOrbit(self, N, Norbit=None, t_MJD=None, psi_deg=None, verbose=0, returnMeanAnomaly=0, returnTrueAnomaly=0 ):
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

        # #***********SYSTEM*PARAMETERS**************************************************
       
    
        m2_MS = self.m2_MJ * MJ_kg/MS_kg # #companion mass in units of SOLAR mass
        #         gamma_ms = 0. #systemic velocity / m s^-1
        d_pc  = 1./ (self.plx_mas/1000.)
    
        if verbose:
            print("%s " % "++++++++++++++++++++")
            print("Primary   mass = %1.3f Msol \t = %4.3f Mjup " % (self.m1_MS, self.m1_MS*MS_kg/MJ_kg))
            print("Secondary mass = %1.3f Msol \t = %4.3f Mjup \t = %4.3f MEarth " % ( m2_MS, self.m2_MJ, self.m2_MJ*MJ_kg/ME_kg))
            print("Inclination  %1.3f deg " % self.i_deg)
            print("Mass ratio q = %4.6f  " %( m2_MS/self.m1_MS))
            print("Period is   %3.1f day \t Eccentricity = %2.1f " % (self.P_day,self.ecc))
            print("Distance is %3.1f pc \t Parallax = %3.1f mas " % (d_pc, self.plx_mas))
            print("omega = %2.1f deg, OMEGA = %2.1f deg, T0 = %2.1f day " % (self.omega_deg, self.OMEGA_deg,self.Tp_day))

        omega_rad = np.deg2rad(self.omega_deg)
        OMEGA_rad = np.deg2rad(self.OMEGA_deg)
        i_rad =     np.deg2rad(self.i_deg)
    
        #*************SIMULATION*PARAMATERS*********************************************
        if Norbit is not None:
            t_day = np.linspace(0, self.P_day*Norbit, N) + self.Tref_MJD
        elif t_MJD is not None:
            t_day = t_MJD
            N = len(t_MJD)

        # R_rv = 50   #number of RV observations
        # R_aric = 50   #number of astrometric observations
        # gnoise_rv = 0.5  # noise level on RV in m/s RMS
        # s_rv = 10.       # error bar on RV measurement in m/s
        # gnoise_aric = 0.05  # noise level on astrometry in mas RMS
        # s_aric = 0.1       # error bar on astromeric measurement in mas 
        # t_day = span(0,P_day*Norbit,N) + Tp_day   # time vector
    
        #**************RADIAL*VELOCITY**************************************************
    
        E_rad = eccentric_anomaly(self.ecc, t_day, self.Tp_day, self.P_day) # eccentric anomaly
        M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        #M = G * ( m1_MS*MS + m2_MJ*MJ ) #relative orbit
        a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
        a_AU = a_m / AU_m #  in AU
        
        if 0==1:        
            THETA_rad = 2*np.arctan( np.sqrt( (1+self.ecc)/(1-self.ecc) ) * np.tan( E_rad/2 ) ) #position angle between radius vector and ref
            THETA_rad = np.arctan2( np.cos(THETA_rad), np.sin(THETA_rad) )
            
            k1 = 2. * np.pi * a_m * np.sin(i_rad) / ( self.P_day*day2sec * (1.-self.ecc**2)**(1./2.) ) #RV semiamplitude
            rv_ms = k1 * ( np.cos( THETA_rad + omega_rad ) + self.ecc*np.cos(omega_rad) ) + self.gamma_ms #radial velocity in m/s.

        else: # damien's method
            THETA_rad = TrueAnomaly(self.ecc,E_rad)
            k1 = 2. * np.pi * a_m * np.sin(i_rad) / ( self.P_day*day2sec * (1.-self.ecc**2)**(1./2.) ) #RV semiamplitude            
            a_mps = RadialVelocitiesConstants(k1,omega_rad,self.ecc)
#             print(a_mps 
            rv_ms = RadialVelocitiesKepler(a_mps[0],a_mps[1],a_mps[2],THETA_rad) + self.gamma_ms
            
        if self.rvLinearDrift_mspyr is not None:
            drift_ms = (t_day - self.Tref_MJD)/year2day * self.rvLinearDrift_mspyr                      
            rv_ms += drift_ms 

        if self.rvQuadraticDrift_mspyr is not None:
            drift_ms = ((t_day - self.Tref_MJD)/year2day)**2 * self.rvQuadraticDrift_mspyr                                            
            rv_ms += drift_ms 

        if self.rvCubicDrift_mspyr is not None:
            drift_ms = ((t_day - self.Tref_MJD)/year2day)**3 * self.rvCubicDrift_mspyr                                            
            rv_ms += drift_ms 
            
            
            
        a_rel_AU = (Ggrav*(self.m1_MS*MS_kg+self.m2_MJ*MJ_kg) / 4. /(np.pi**2.) *(self.P_day*day2sec)**2.)**(1./3.)/AU_m
    
        # print('i_deg = %3.2f, om_deg = %3.2f, k1_mps = %3.2f, phi0 = %3.2f, t_day[0] = %3.2f, rv[0] = %3.2f, THETA_rad[0] = %3.2f, E_rad[0] = %2.2f' % (self.i_deg,self.omega_deg,k1, self.Tp_day/self.P_day, t_day[0],rv_ms[0], THETA_rad[0],E_rad[0])

        if verbose == 1:    
            print("Astrometric semimajor axis of Primary: a = %3.3f AU \t %6.3f muas " % (a_AU,a_AU/d_pc*1.e6))
            print("Relative semimajor axis of Primary: a = %3.3f AU \t %6.2f mas " %(a_rel_AU,a_rel_AU/d_pc*1.e3))
            print("Radial velocity semi-amplitude: K1 =  %4.2f m/s  " % k1)
        
        #**************ASTROMETRY********************************************************
        # a_rad = a_m / (d_pc*pc_m) #for small angles
        a_rad = np.arctan2(a_m,d_pc*pc_m)
        # print a_rad, a_rad2
        # 1/0
        a_mas = a_rad * rad2mas # semimajor axis in mas
 
        aRel_mas = np.arctan2(a_rel_AU*AU_m,d_pc*pc_m) * rad2mas # relative semimajor axis in mas
#         print 'aRel_mas = %3.3f mas' % aRel_mas
        TIC     = pjGet_TIC( [ a_mas   , self.omega_deg     , self.OMEGA_deg, self.i_deg ] ) #Thiele-Innes constants
        TIC_rel = pjGet_TIC( [ aRel_mas, self.omega_deg+180., self.OMEGA_deg, self.i_deg ] ) #Thiele-Innes constants
        # A = TIC[0] B = TIC[1] F = TIC[2] G = TIC[3]
        
        if psi_deg is not None:
            # psi_rad = np.deg2rad(psi_deg)
            phi1 = astrom_signal(t_day, psi_deg, self.ecc, self.P_day, self.Tp_day, TIC)
            phi1_rel = astrom_signal(t_day, psi_deg, self.ecc, self.P_day, self.Tp_day, TIC_rel)
            phi2 = np.nan
            phi2_rel = np.nan
            
        else:
            #first baseline  second baseline
            bstart1 = 0.
            bstart2 = 90.    #baseline offset in deg
#             bspread1 = 0.    bspread2 = 0.    #baseline spread around offset in deg
    
            # for FORS aric + CRIRES RV simulation, the aric measurement gives both axis simultaneously    
            psi_deg1 = np.ones(N)*bstart1# array(bstart1,N)
            # psi_rad1 = psi_deg1*deg2rad
            psi_deg2 = np.ones(N)*bstart2
            # psi_rad2 = psi_deg2*deg2rad
        
            phi1 = astrom_signal(t_day, psi_deg1, self.ecc, self.P_day, self.Tp_day, TIC)
            phi2 = astrom_signal(t_day, psi_deg2, self.ecc, self.P_day, self.Tp_day, TIC)
            phi1_rel = astrom_signal(t_day, psi_deg1, self.ecc, self.P_day, self.Tp_day, TIC_rel)
            phi2_rel = astrom_signal(t_day, psi_deg2, self.ecc, self.P_day, self.Tp_day, TIC_rel)
    
        if returnMeanAnomaly:
            M_rad = mean_anomaly(t_day, self.Tp_day, self.P_day)
            return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel, M_rad]
            
        elif returnTrueAnomaly:
#           M_rad = mean_anomaly(t_day,self.Tp_day,self.P_day)
            return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel, THETA_rad, TIC_rel]
            
        return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel]


    # def pjGetRV(self,t_day):
    def compute_radial_velocity(self, t_day, component='primary'):
        """Compute radial velocity of primary or secondary component.

         updated: J. Sahlmann   25.01.2016   STScI/ESA
         updated: J. Sahlmann   13.07.2018   STScI/AURA

        Parameters
        ----------
        t_day
        component

        Returns
        -------

        """

        # m2_MS = self.m2_MJ * MJ_kg/MS_kg# #companion mass in units of SOLAR mass
        i_rad =     np.deg2rad(self.i_deg)
    
        #**************RADIAL*VELOCITY**************************************************    
        E_rad = eccentric_anomaly(self.ecc, t_day, self.Tp_day, self.P_day) # eccentric anomaly
        if component=='primary':
            M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
            omega_rad = np.deg2rad(self.omega_deg)
        elif component == 'secondary':
            M = Ggrav * (self.m1_MS * MS_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the secondary mass
            omega_rad = np.deg2rad(self.omega_deg + 180.)

        a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the component mass in m
        # a_AU = a_m / AU_m #  in AU
        
        # damien's method
        THETA_rad = TrueAnomaly(self.ecc,E_rad)
        k_m = 2. * np.pi * a_m * np.sin(i_rad) / ( self.P_day*day2sec * (1.-self.ecc**2)**(1./2.) ) #RV semiamplitude
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


    def plot_rv_orbit(self, component='primary', n_curve=100, n_orbit=1, line_color='k', line_style='-', line_width=1, rv_factor=1., time_offset_day=0.):
        """Plot the radial velocity orbit of the primary

        Returns
        -------

        """
        t_day = np.linspace(0, self.P_day * n_orbit, n_curve) - self.P_day/2 + self.Tp_day + time_offset_day
        t_plot = Time(t_day, format='mjd').jyear
        if component=='primary':
            rv_mps = self.compute_radial_velocity(t_day, component=component) * rv_factor
            pl.plot(t_plot, rv_mps, ls=line_style, color=line_color, lw=line_width)
        elif component=='secondary':
            rv_mps = self.compute_radial_velocity(t_day, component=component) * rv_factor
            pl.plot(t_plot, rv_mps, ls=line_style, color=line_color, lw=line_width)
        elif component=='both':
            rv_mps_1 = self.compute_radial_velocity(t_day, component='primary') * rv_factor
            rv_mps_2 = self.compute_radial_velocity(t_day, component='secondary') * rv_factor
            pl.plot(t_plot, rv_mps_1, ls=line_style, color=line_color, lw=line_width+2, label='primary')
            pl.plot(t_plot, rv_mps_2, ls=line_style, color=line_color, lw=line_width, label='secondary')
        elif component=='difference':
            rv_mps_1 = self.compute_radial_velocity(t_day, component='primary') * rv_factor
            rv_mps_2 = self.compute_radial_velocity(t_day, component='secondary') * rv_factor
            pl.plot(t_plot, rv_mps_1-rv_mps_2, ls=line_style, color=line_color, lw=line_width+2, label='difference')


    
    def pjGetOrbitFast(self, N, Norbit=None, t_MJD=None, psi_deg=None, verbose=0):
    # /* DOCUMENT ARV -- simulate fast 1D astrometry for planet detection limits
    #    written: J. Sahlmann   18 May 2015   ESAC
    # */
               
    
        m2_MS = self.m2_MJ * MJ_kg/MS_kg# #companion mass in units of SOLAR mass 
        d_pc  = 1./ (self.plx_mas/1000.)    

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
        
        TIC     = pjGet_TIC( [ a_mas   , self.omega_deg     , self.OMEGA_deg, self.i_deg ] ) #Thiele-Innes constants
        
        phi1 = astrom_signal(t_day, psi_deg, self.ecc, self.P_day, self.Tp_day, TIC)
        phi1_rel = np.nan #astrom_signal(t_day,psi_deg,self.ecc,self.P_day,self.Tp_day,TIC_rel)
        phi2 = np.nan
        phi2_rel = np.nan
        rv_ms=np.nan
    
        return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel]


    def pjGetBarycentricAstrometricOrbitFast(self, t_MJD, spsi, cpsi):#, scan_angle_definition='hipparcos'):
        # /* DOCUMENT ARV -- simulate fast 1D astrometry for planet detection limits
        #    written: J. Sahlmann   18 May 2015   ESAC
        #    updated: J. Sahlmann   25.01.2016   STScI/ESA
        # */
        #       #**************ASTROMETRY********************************************************
        #       M = Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        #       a_m = ( M / (4. * np.pi**2.) * (P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
        #       a_rad = np.arctan2(a_m,d_pc*pc_m)
        #       a_mas = a_rad * rad2mas # semimajor axis in mas 
        #       TIC     = pjGet_TIC( [ a_mas   , omega_deg     , OMEGA_deg, i_deg ] ) #Thiele-Innes constants
        #       phi1 = astrom_signalFast(t_MJD,spsi,cpsi,ecc,P_day,Tp_day,TIC)
        #       return phi1
               
#         t_MJD = self.MjdUsedInTcspsi
    
        # m2_MS = self.m2_MJ * MJ_kg/MS_kg# #companion mass in units of SOLAR mass
        d_pc  = 1./ (self.plx_mas/1000.)    

        # omega_rad = np.deg2rad(self.omega_deg)
        # OMEGA_rad = np.deg2rad(self.OMEGA_deg)
        # i_rad = np.deg2rad(self.i_deg)
    
        #**************ASTROMETRY********************************************************
        
        M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m        
        a_rad = np.arctan2(a_m,d_pc*pc_m)
        a_mas = a_rad * rad2mas # semimajor axis in mas         
        TIC     = pjGet_TIC( [ a_mas   , self.omega_deg     , self.OMEGA_deg, self.i_deg ] ) #Thiele-Innes constants
        phi1 = astrom_signalFast(t_MJD, spsi, cpsi, self.ecc, self.P_day, self.Tp_day, TIC, scan_angle_definition=self.scan_angle_definition)
        # phi1 = astrom_signalFast(t_MJD, spsi, cpsi, self.ecc, self.P_day, self.Tp_day, TIC)
        return phi1


    def relative_orbit_fast(self, t_MJD, spsi, cpsi, unit='mas', shift_omega_by_pi=True, coordinate_system='cartesian'):
        '''
        Simulate fast 1D orbital astrometry
        written: J. Sahlmann   18 May 2015   ESAC
        updated: J. Sahlmann   25.01.2016   STScI/ESA
        updated: J. Sahlmann   27 February 2017   STScI/AURA

        returns relative orbit in linear or angular units       
        '''
    
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
            d_pc  = 1./ (self.plx_mas/1000.)    
            a_rad = np.arctan2(a_rel_m,d_pc*pc_m)
            # semimajor axis in mas         
            a_rel_mas = a_rad * rad2mas 
            a_rel = a_rel_mas
        elif unit == 'meter':
            a_rel = a_rel_m     
            
        #Thiele-Innes constants 
        TIC = pjGet_TIC([a_rel, omega_rel_deg, self.OMEGA_deg, self.i_deg])
        
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

    
        
    
    
    
#     def pjGetParameters(self, verbose = 0):
#         m2_MS = self.m2_MJ * MJ_kg/MS_kg# #companion mass in units of SOLAR mass
# #         gamma_ms = 0. #systemic velocity / m s^-1
#         d_pc  = 1./ (self.plx_mas/1000.)
#         if verbose:
#             print("%s " % "++++++++++++++++++++")
#             print("Primary   mass = %1.3f Msol \t = %4.3f Mjup " % (self.m1_MS, self.m1_MS*MS_kg/MJ_kg))
#             print("Secondary mass = %1.3f Msol \t = %4.3f Mjup \t = %4.3f MEarth " % ( m2_MS, self.m2_MJ, self.m2_MJ*MJ_kg/ME_kg))
#             print("Inclination  %1.3f deg " % self.i_deg)
#             print("Mass ratio q = %4.6f  " %( m2_MS/self.m1_MS))
#             print("Period is   %3.1f day \t Eccentricity = %2.1f " % (self.P_day,self.ecc))
#             print("Distance is %3.1f pc \t Parallax = %3.1f mas " % (d_pc, self.plx_mas))
#             print("omega = %2.1f deg, OMEGA = %2.1f deg, T0 = %2.1f day " % (self.omega_deg, self.OMEGA_deg,self.Tp_day))
#
#         omega_rad = np.deg2rad(self.omega_deg)
#         OMEGA_rad = np.deg2rad(self.OMEGA_deg)
#         i_rad     = np.deg2rad(self.i_deg)
#
#         #*************SIMULATION*PARAMATERS*********************************************
#         # Norbit = 2.   #number of orbits covered
#         # t_day = np.linspace(0,self.P_day*Norbit,N) + self.Tp_day
#
#         # R_rv = 50   #number of RV observations
#         # R_aric = 50   #number of astrometric observations
#         # gnoise_rv = 0.5  # noise level on RV in m/s RMS
#         # s_rv = 10.       # error bar on RV measurement in m/s
#         # gnoise_aric = 0.05  # noise level on astrometry in mas RMS
#         # s_aric = 0.1       # error bar on astromeric measurement in mas
#         # t_day = span(0,P_day*Norbit,N) + Tp_day   # time vector
#
#
#         #**************RADIAL*VELOCITY**************************************************
#
#         # E_rad = eccentric_anomaly(self.ecc,t_day,self.Tp_day,self.P_day) # eccentric anomaly
#         # THETA_rad = 2*np.arctan( np.sqrt( (1+self.ecc)/(1-self.ecc) ) * np.tan( E_rad/2 ) ) #position angle between radius vector and ref
#         M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
#         #M = G * ( m1_MS*MS + m2_MJ*MJ ) #relative orbit
#
#         a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
#         a_AU = a_m / AU_m #  in AU
#
#         k1 = 2. * np.pi * a_m * np.sin(i_rad) / ( self.P_day*day2sec * (1.-self.ecc**2)**(1./2.) ) #RV semiamplitude
#         # rv_ms = k1 * ( np.cos( THETA_rad + omega_rad ) + self.ecc*np.cos(omega_rad) ) + gamma_ms #radial velocity in m/s.
#
#         a_rel_AU = (Ggrav*(self.m1_MS*MS_kg+self.m2_MJ*MJ_kg) / 4. /(np.pi**2.) *(self.P_day*day2sec)**2.)**(1./3.)/AU_m
#         self.a_rel_AU = a_rel_AU
#         self.a1_mas = a_AU/d_pc*1.e3
#         if verbose:
#             print("Astrometric semimajor axis of Primary: a = %3.3f AU \t %6.3f muas " % (a_AU,a_AU/d_pc*1.e6))
#             print("Relative semimajor axis of Primary: a = %3.3f AU \t %6.2f mas " %(a_rel_AU,a_rel_AU/d_pc*1.e3))
#             print("Radial velocity semi-amplitude: K1 =  %4.2f m/s  " % k1)
#
#         #**************ASTROMETRY********************************************************
#         a_rad = a_m / (d_pc*pc_m) #for small angles
#         a_mas = a_rad * rad2mas # semimajor axis in mas
#
#         TIC = pjGet_TIC( [ a_mas, self.omega_deg, self.OMEGA_deg, self.i_deg ] ) #Thiele-Innes constants
#         A = TIC[0] B = TIC[1] F = TIC[2] G = TIC[3]
#
#
#         #first baseline  second baseline
#         # bstart1 = 0.     bstart2 = 90.    #baseline offset in deg
#         # bspread1 = 0.    bspread2 = 0.    #baseline spread around offset in deg
#
#         # # for FORS aric + CRIRES RV simulation, the aric measurement gives both axis simultaneously
#         # psi_deg1 = np.ones(N)*bstart1# array(bstart1,N)
#         # psi_rad1 = psi_deg1*deg2rad
#         # psi_deg2 = np.ones(N)*bstart2
#         # psi_rad2 = psi_deg2*deg2rad
#
#         # phi1 = astrom_signal(t_day,psi_deg1,self.ecc,self.P_day,self.Tp_day,TIC)
#         # phi2 = astrom_signal(t_day,psi_deg2,self.ecc,self.P_day,self.Tp_day,TIC)
    
    

#     def getPpm(self,t_MJD,psi_deg=None, offsetRA_mas=0, offsetDE_mas=0, tref_MJD=None, externalParallaxFactors=None, horizons_file_seed=None):
    def ppm(self, t_MJD, psi_deg=None, offsetRA_mas=0, offsetDE_mas=0, externalParallaxFactors=None, horizons_file_seed=None, instrument=None, verbose=0):
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
            parf = getParallaxFactors(self.RA_deg, self.DE_deg, t_JD, horizons_file_seed=horizons_file_seed, verbose=verbose, instrument=instrument)
                        
        self.parf = parf
        if self.Tref_MJD is not None:
            t0_MJD = self.Tref_MJD
        else:    
            t0_MJD = np.mean(t_MJD)
        trel_year = (t_MJD - t0_MJD)/year2day

        self.t0_MJD = t0_MJD
        # % sin(psi) and cos(psi)
        if psi_deg is not None:
            psi_rad = np.deg2rad(psi_deg)
            spsi = np.sin(psi_rad)
            cpsi = np.cos(psi_rad)
            t = trel_year 
        else:
            t, cpsi, spsi, xi, yi = get_spsi_cpsi_for_2Dastrometry(trel_year, scan_angle_definition=self.scan_angle_definition)
            # tmp = np.arange(1,2*Nframes+1)
            # spsi = (np.arange(1,2*Nframes+1)+1)%2# % first X then Y
            # cpsi = (np.arange(1,2*Nframes+1)  )%2
            # t = np.sort(np.tile(trel_year ,2))
            
        tspsi = t*spsi
        tcpsi = t*cpsi
        
        if psi_deg is not None:
            if externalParallaxFactors is None:
                ppfact = parf[0] * cpsi + parf[1] * spsi    # see Sahlmann+11 Eq. 1 / 8
            else:
                ppfact = parf
        else:
            # xi = spsi==0    #index of X coordinates (cpsi = 1) psi =  0 deg
            # yi = cpsi==0    #index of Y coordinates (spsi = 1) psi = 90 deg
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

        inVec = np.array([offsetRA_mas, offsetDE_mas, self.plx_mas, self.muRA_mas, self.muDE_mas])
        print(inVec)
        ppm = np.dot(C.T, inVec)
        self.ppm = ppm
        if psi_deg is not None:
            return ppm
        else:
            ppm2d = [ppm[xi],ppm[yi]]
            return ppm2d


    def plot_orbits(self, timestamps_curve_2D=None, timestamps_probe_2D=None, timestamps_probe_2D_label=None,
                    delta_mag=None, N_orbit=1., N_curve=100., save_plot=False, plot_dir=None,
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
        if timestamps_curve_2D is None:
            timestamps_curve_2D = np.linspace(self.Tp_day - self.P_day, self.Tp_day + N_orbit + self.P_day, N_curve)

        timestamps_curve_1D, cpsi_curve, spsi_curve, xi_curve, yi_curve = get_spsi_cpsi_for_2Dastrometry( timestamps_curve_2D )
        # relative orbit
        phi0_curve_relative = self.relative_orbit_fast(timestamps_curve_1D, spsi_curve, cpsi_curve, shift_omega_by_pi = True)

        if timestamps_probe_2D is not None:   
            timestamps_probe_1D, cpsi_probe, spsi_probe, xi_probe, yi_probe = get_spsi_cpsi_for_2Dastrometry( timestamps_probe_2D )
            phi0_probe_relative = self.relative_orbit_fast(timestamps_probe_1D, spsi_probe, cpsi_probe, shift_omega_by_pi = True)
    
        if delta_mag is not None:        
            # fractional luminosity
            beta = fractional_luminosity( 0. , 0.+delta_mag )
            #     fractional mass
            f = getB(self.m1_MS, self.m2_MS)
    
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
        axes[0].plot(phi0_curve_barycentre[xi_curve] ,phi0_curve_barycentre[yi_curve],'k--',lw=line_width, color=line_color, ls=line_style) #, label='Barycentre'
        # plot individual epochs    
        if timestamps_probe_2D is not None:
            axes[0].plot(phi0_probe_barycentre[xi_probe],phi0_probe_barycentre[yi_probe],'bo',mfc='0.7', label=timestamps_probe_2D_label)

        if delta_mag is not None:
            axes[0].plot(phi0_curve_photocentre[xi_curve],phi0_curve_photocentre[yi_curve],'k-',lw=1, label='Photocentre')
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
                    delta_mag=None, N_orbit=1., N_curve=100., save_plot=False, plot_dir=None,
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
            t_plot_probe = getattr(Time(timestamps_probe_2D, format='mjd'), timeformat)

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
                axes[0].plot(t_plot_probe, ppm_probe_mas[0], 'bo', label=timestamps_probe_2D_label, **kwargs)
                axes[1].plot(t_plot_probe, ppm_probe_mas[1], 'bo', label=timestamps_probe_2D_label, **kwargs)
                if timestamps_probe_2D_label is not None:
                    axes[0].legend(loc='best')

            pl.show()
            if save_plot:
                fig_name = os.path.join(plot_dir, '{}_ppm_time.pdf'.format(name_seed))
                plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

        # return
        # # 1/0
        #
        #
        # if delta_mag is not None:
        #     axes[0].plot(phi0_curve_photocentre[xi_curve], phi0_curve_photocentre[yi_curve], 'k-',
        #                  lw=1, label='Photocentre')
        #     if timestamps_probe_2D is not None:
        #         axes[0].plot(phi0_probe_photocentre[xi_probe], phi0_probe_photocentre[yi_probe],
        #                      'bo')
        #
        # if show_orientation:
        #     # arrow_index_1 = np.int(N_curve/3.3)
        #     arrow_index_1 = np.int(N_curve / 5)
        #     arrow_index_2 = arrow_index_1 + 10
        #     length_factor = 10
        #     arrow_factor = 1000
        #
        #     # ax = pl.axes()
        #     arrow_base_x = phi0_curve_barycentre[xi_curve][arrow_index_1]
        #     arrow_base_y = phi0_curve_barycentre[yi_curve][arrow_index_1]
        #     arrow_delta_x = phi0_curve_barycentre[xi_curve][arrow_index_2] - arrow_base_x
        #     arrow_delta_y = phi0_curve_barycentre[yi_curve][arrow_index_2] - arrow_base_y
        #
        #     axes[0].arrow(arrow_base_x + arrow_offset_x, arrow_base_y,
        #                   arrow_delta_x * length_factor, arrow_delta_y * length_factor,
        #                   head_width=0.05 * arrow_factor, head_length=0.1 * arrow_factor,
        #                   fc=line_color, ec=line_color)  # , head_width=0.05, head_length=0.1
        #
        # # plot origin = position of barycentre
        # axes[0].plot(0, 0, 'kx')
        # axes[0].axhline(y=0, color='0.7', ls='--', zorder=-50)
        # axes[0].axvline(x=0, color='0.7', ls='--', zorder=-50)
        #
        # axes[0].set_xlabel('Offset in Right Ascension (mas)')
        # axes[0].set_ylabel('Offset in Declination (mas)')
        # axes[0].axis('equal')
        # if invert_xaxis:
        #     axes[0].invert_xaxis()
        # axes[0].legend(loc='best')
        # axes[0].set_title('Bary-/photocentric orbit of M1')
        #
        # # second panel
        # # plot smooth orbit curve
        # axes[1].plot(phi0_curve_relative[xi_curve], phi0_curve_relative[yi_curve], 'k-',
        #              lw=line_width, color=line_color, ls=line_style)
        # # plot individual epochs
        # if timestamps_probe_2D is not None:
        #     axes[1].plot(phi0_probe_relative[xi_probe], phi0_probe_relative[yi_probe], 'bo',
        #                  label=timestamps_probe_2D_label)
        #     print('relative separation: {}'.format(
        #         np.linalg.norm([phi0_probe_relative[xi_probe], phi0_probe_relative[yi_probe]],
        #                        axis=0)))
        #
        # if show_orientation:
        #     # ax = pl.axes()
        #     arrow_base_x = phi0_curve_relative[xi_curve][arrow_index_1]
        #     arrow_base_y = phi0_curve_relative[yi_curve][arrow_index_1]
        #     arrow_delta_x = phi0_curve_relative[xi_curve][arrow_index_2] - arrow_base_x
        #     arrow_delta_y = phi0_curve_relative[yi_curve][arrow_index_2] - arrow_base_y
        #
        #     axes[1].arrow(arrow_base_x + arrow_offset_x, arrow_base_y,
        #                   arrow_delta_x * length_factor, arrow_delta_y * length_factor,
        #                   head_width=0.05 * arrow_factor, head_length=0.1 * arrow_factor,
        #                   fc=line_color, ec=line_color)
        #
        # # plot origin = position of primary
        # axes[1].plot(0, 0, 'kx')
        # axes[1].axhline(y=0, color='0.7', ls='--', zorder=-50)
        # axes[1].axvline(x=0, color='0.7', ls='--', zorder=-50)
        #
        # axes[1].set_xlabel('Offset in Right Ascension (mas)')
        # axes[1].axis('equal')
        # axes[1].legend(loc='best')
        # axes[1].set_title('Relative orbit of M2 about M1')
        # if (not axes[1]._sharex) and (invert_xaxis):
        #     axes[1].invert_xaxis()
        # pl.show()
        # if save_plot:
        #     figName = os.path.join(plot_dir, 'astrometric_orbits.pdf')
        #     plt.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0.05)
        #
        #
        #
        #
        # timestamps_curve_1D, cpsi_curve, spsi_curve, xi_curve, yi_curve = get_spsi_cpsi_for_2Dastrometry(
        #     timestamps_curve_2D)
        # # relative orbit
        # phi0_curve_relative = self.relative_orbit_fast(timestamps_curve_1D, spsi_curve, cpsi_curve,
        #                                                shift_omega_by_pi=True)
        #
        # if timestamps_probe_2D is not None:
        #     timestamps_probe_1D, cpsi_probe, spsi_probe, xi_probe, yi_probe = get_spsi_cpsi_for_2Dastrometry(
        #         timestamps_probe_2D)
        #     phi0_probe_relative = self.relative_orbit_fast(timestamps_probe_1D, spsi_probe,
        #                                                    cpsi_probe, shift_omega_by_pi=True)
        #
        # if delta_mag is not None:
        #     # fractional luminosity
        #     beta = fractional_luminosity(0., 0. + delta_mag)
        #     #     fractional mass
        #     f = getB(self.m1_MS, self.m2_MS)
        #
        #     # photocentre orbit about the system's barycentre
        #     phi0_curve_photocentre = (f - beta) * self.relative_orbit_fast(timestamps_curve_1D,
        #                                                                    spsi_curve, cpsi_curve,
        #                                                                    shift_omega_by_pi=False)
        #     if timestamps_probe_2D is not None:
        #         phi0_probe_photocentre = (f - beta) * self.relative_orbit_fast(timestamps_probe_1D,
        #                                                                        spsi_probe,
        #                                                                        cpsi_probe,
        #                                                                        shift_omega_by_pi=False)
        #
        # # barycentric orbit of M1
        # phi0_curve_barycentre = self.pjGetBarycentricAstrometricOrbitFast(timestamps_curve_1D,
        #                                                                   spsi_curve, cpsi_curve)
        # if timestamps_probe_2D is not None:
        #     phi0_probe_barycentre = self.pjGetBarycentricAstrometricOrbitFast(timestamps_probe_1D,
        #                                                                       spsi_probe,
        #                                                                       cpsi_probe)
        #
        # n_figure_columns = 2
        # n_figure_rows = 1
        # # fig, axes = pl.subplots(n_figure_rows, n_figure_columns, figsize=(n_figure_columns*6, n_figure_rows*5), facecolor='w', edgecolor='k', sharex=True, sharey=True)
        #
        # if new_figure:
        #     fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
        #                             figsize=(n_figure_columns * 6, n_figure_rows * 5),
        #                             facecolor='w', edgecolor='k', sharex=share_axes,
        #                             sharey=share_axes)
        # else:
        #     axes = pl.gcf().axes
        # # plot smooth orbit curve
        # axes[0].plot(phi0_curve_barycentre[xi_curve], phi0_curve_barycentre[yi_curve], 'k--',
        #              lw=line_width, color=line_color, ls=line_style)  # , label='Barycentre'
        # # plot individual epochs
        # if timestamps_probe_2D is not None:
        #     axes[0].plot(phi0_probe_barycentre[xi_probe], phi0_probe_barycentre[yi_probe], 'bo',
        #                  mfc='0.7', label=timestamps_probe_2D_label)
        #
        # if delta_mag is not None:
        #     axes[0].plot(phi0_curve_photocentre[xi_curve], phi0_curve_photocentre[yi_curve], 'k-',
        #                  lw=1, label='Photocentre')
        #     if timestamps_probe_2D is not None:
        #         axes[0].plot(phi0_probe_photocentre[xi_probe], phi0_probe_photocentre[yi_probe],
        #                      'bo')
        #
        # if show_orientation:
        #     # arrow_index_1 = np.int(N_curve/3.3)
        #     arrow_index_1 = np.int(N_curve / 5)
        #     arrow_index_2 = arrow_index_1 + 10
        #     length_factor = 10
        #     arrow_factor = 1000
        #
        #     # ax = pl.axes()
        #     arrow_base_x = phi0_curve_barycentre[xi_curve][arrow_index_1]
        #     arrow_base_y = phi0_curve_barycentre[yi_curve][arrow_index_1]
        #     arrow_delta_x = phi0_curve_barycentre[xi_curve][arrow_index_2] - arrow_base_x
        #     arrow_delta_y = phi0_curve_barycentre[yi_curve][arrow_index_2] - arrow_base_y
        #
        #     axes[0].arrow(arrow_base_x + arrow_offset_x, arrow_base_y,
        #                   arrow_delta_x * length_factor, arrow_delta_y * length_factor,
        #                   head_width=0.05 * arrow_factor, head_length=0.1 * arrow_factor,
        #                   fc=line_color, ec=line_color)  # , head_width=0.05, head_length=0.1
        #
        # # plot origin = position of barycentre
        # axes[0].plot(0, 0, 'kx')
        # axes[0].axhline(y=0, color='0.7', ls='--', zorder=-50)
        # axes[0].axvline(x=0, color='0.7', ls='--', zorder=-50)
        #
        # axes[0].set_xlabel('Offset in Right Ascension (mas)')
        # axes[0].set_ylabel('Offset in Declination (mas)')
        # axes[0].axis('equal')
        # if invert_xaxis:
        #     axes[0].invert_xaxis()
        # axes[0].legend(loc='best')
        # axes[0].set_title('Bary-/photocentric orbit of M1')
        #
        # # second panel
        # # plot smooth orbit curve
        # axes[1].plot(phi0_curve_relative[xi_curve], phi0_curve_relative[yi_curve], 'k-',
        #              lw=line_width, color=line_color, ls=line_style)
        # # plot individual epochs
        # if timestamps_probe_2D is not None:
        #     axes[1].plot(phi0_probe_relative[xi_probe], phi0_probe_relative[yi_probe], 'bo',
        #                  label=timestamps_probe_2D_label)
        #     print('relative separation: {}'.format(
        #         np.linalg.norm([phi0_probe_relative[xi_probe], phi0_probe_relative[yi_probe]],
        #                        axis=0)))
        #
        # if show_orientation:
        #     # ax = pl.axes()
        #     arrow_base_x = phi0_curve_relative[xi_curve][arrow_index_1]
        #     arrow_base_y = phi0_curve_relative[yi_curve][arrow_index_1]
        #     arrow_delta_x = phi0_curve_relative[xi_curve][arrow_index_2] - arrow_base_x
        #     arrow_delta_y = phi0_curve_relative[yi_curve][arrow_index_2] - arrow_base_y
        #
        #     axes[1].arrow(arrow_base_x + arrow_offset_x, arrow_base_y,
        #                   arrow_delta_x * length_factor, arrow_delta_y * length_factor,
        #                   head_width=0.05 * arrow_factor, head_length=0.1 * arrow_factor,
        #                   fc=line_color, ec=line_color)
        #
        # # plot origin = position of primary
        # axes[1].plot(0, 0, 'kx')
        # axes[1].axhline(y=0, color='0.7', ls='--', zorder=-50)
        # axes[1].axvline(x=0, color='0.7', ls='--', zorder=-50)
        #
        # axes[1].set_xlabel('Offset in Right Ascension (mas)')
        # axes[1].axis('equal')
        # axes[1].legend(loc='best')
        # axes[1].set_title('Relative orbit of M2 about M1')
        # if (not axes[1]._sharex) and (invert_xaxis):
        #     axes[1].invert_xaxis()
        # pl.show()
        # if save_plot:
        #     figName = os.path.join(plot_dir, 'astrometric_orbits.pdf')
        #     plt.savefig(figName, transparent=True, bbox_inches='tight', pad_inches=0.05)
        #
        # # show barycentric offsets as function of time
        # if show_time:
        #     n_figure_columns = 2
        #     n_figure_rows = 1
        #     fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
        #                             figsize=(n_figure_columns * 6, n_figure_rows * 5),
        #                             facecolor='w', edgecolor='k', sharex=share_axes,
        #                             sharey=share_axes)
        #     # plot smooth orbit curve
        #     axes[0].plot(Time(timestamps_curve_1D[xi_curve], format='mjd').jyear,
        #                  phi0_curve_barycentre[xi_curve], 'k-',
        #                  lw=line_width, color=line_color, ls=line_style)
        #     axes[1].plot(Time(timestamps_curve_1D[yi_curve], format='mjd').jyear,
        #                  phi0_curve_barycentre[yi_curve], 'k-',
        #                  lw=line_width, color=line_color, ls=line_style)
        #
        #     # plot individual epochs
        #     if timestamps_probe_2D is not None:
        #         axes[0].plot(Time(timestamps_probe_1D[xi_probe], format='mjd').jyear,
        #                      phi0_probe_barycentre[xi_probe], 'bo',
        #                      mfc='0.7', label=timestamps_probe_2D_label)
        #         axes[1].plot(Time(timestamps_probe_1D[yi_probe], format='mjd').jyear,
        #                      phi0_probe_barycentre[yi_probe], 'bo',
        #                      mfc='0.7', label=timestamps_probe_2D_label)
        #
        #     pl.show()


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
                # 1/0

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
            orb = OrbitSystem(P_day=1., ecc=0.0, m1_MS=1.0, m2_MJ=0.0, omega_deg=0., OMEGA_deg=0., i_deg=0., Tp_day=0,
                              RA_deg=self.RA_deg, DE_deg=self.DE_deg, plx_mas=self.p[2], muRA_mas=self.p[3],
                              muDE_mas=self.p[4], Tref_MJD=self.tref_MJD)
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
                ppm_curve = orb.ppm(self.tmodel_MJD, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                                    horizons_file_seed=horizons_file_seed, psi_deg=psi_deg)
            ppm_meas = orb.ppm(self.t_MJD_epoch, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                               horizons_file_seed=horizons_file_seed, psi_deg=psi_deg)
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
            pl.legend()

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
                # print('Setting {} to {}'.format(key, value))
                print('Setting {}'.format(key))

                # set defaults
            default_dict = {'outlier_sigma_threshold': 3.,
                            'absolute_threshold': 10.,
                            'residuals': None,
                            'scan_angle_definition': 'hipparcos'
                            }

            for key, value in default_dict.items():
                if key not in attribute_dict.keys():
                    setattr(self, key, value)

        linear_coefficient_matrix = self.linear_coefficients['matrix']

        number_of_companions = len(self.model_parameters)

        self.number_of_companions = number_of_companions
        model_name = 'k{:d}'.format(number_of_companions)

        T = self.data.epoch_data

        # parameters of first companion
        theta_0 = self.model_parameters[0]
        theta_names = theta_0.keys()

        if ('plx_abs_mas' in theta_names) & ('plx_corr_mas' in theta_names):
            theta_0['plx_mas ']= theta_0['plx_abs_mas'] + ['plx_corr_mas']

        # compute positions at measurement dates according to best-fit model p (no dcr)
        ppm_parameters = np.array([theta_0['offset_alphastar_mas'], theta_0['offset_delta_mas'],
                          theta_0['plx_mas'], theta_0['muRA_mas'], theta_0['muDE_mas']])

        self.ppm_model = np.array(np.dot(linear_coefficient_matrix[0:len(ppm_parameters), :].T, ppm_parameters)).flatten()

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
        else:
            dcr = np.zeros(linear_coefficient_matrix.shape[1])
        self.DCR = dcr

        for p in range(number_of_companions):
            theta_p = self.model_parameters[p]
            if 'm2_MS' in theta_names:
                theta_p['m2_MJ'] = theta_p['m2_MS'] * MS_kg / MJ_kg

            tmporb = OrbitSystem(attribute_dict=theta_p)
            orbit_model = tmporb.pjGetBarycentricAstrometricOrbitFast(np.array(T['MJD']),
                                                                      np.array(T['spsi']),
                                                                      np.array(T['cpsi']))
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
        self.t_MJD_epoch = np.zeros(len(medi))

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
                # if self.data_type == '2d':
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
            # if self.data_type == '2d':
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
            self.stdResidualX[jj] = np.std(residuals[tmpIndexX])
            # if self.data_type == '2d':
            if '2d' in self.data_type:
                self.DCR_Ymean[jj] = np.average(self.DCR[tmpIndexY])
                self.meanResidualY[jj] = np.average(residuals[tmpIndexY], weights=1. / (T['sigma_da_mas'][tmpIndexY] ** 2.))
                self.parfYmean[jj] = np.average(T['ppfact'][tmpIndexY])
                self.stdResidualY[jj] = np.std(residuals[tmpIndexY])

            # on the fly inter-epoch outlier detection
            outliers = {}
            outliers['x'] = {}
            outliers['x']['index'] = tmpIndexX
            outliers['x']['std_residual'] = self.stdResidualX[jj]
            # if self.data_type == '2d':
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
            # if self.data_type == '2d':
            if '2d' in self.data_type:
                self.errResidualY[jj] = self.stdResidualY[jj] / np.sqrt(len(tmpIndexY))

            # %         from Lazorenko writeup:
            self.x_e_laz[jj] = np.sum(residuals[tmpIndexX] / (T['sigma_da_mas'][tmpIndexX] ** 2.)) / np.sum(
                1 / (T['sigma_da_mas'][tmpIndexX] ** 2.))
            self.sx_star_laz[jj] = 1 / np.sqrt(np.sum(1 / (T['sigma_da_mas'][tmpIndexX] ** 2.)));

            # if self.data_type == '2d':
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
        # elif self.data_type == '2d':
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
        # elif self.data_type == '2d':
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
        # elif self.data_type == '2d':
            self.epoch_omc_std_Y = np.std(self.meanResidualY)
            self.epoch_omc_std = np.std([self.meanResidualX, self.meanResidualY])
            self.epoch_precision_mean = np.mean([self.errResidualX, self.errResidualY])

        self.residuals = residuals

    def print_residual_statistics(self):
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
        # elif self.data_type == '2d':
            # print((np.mean([self.errResidualX, self.errResidualY], axis=0)))
            print('Epoch   precision (x_e_laz)'),
            print(np.mean([self.sx_star_laz, self.sy_star_laz], axis=0))
            print('Average precision (naive) %3.3f mas' % (np.mean([self.errResidualX, self.errResidualY])))
            print('Average precision (x_e_laz) %3.3f mas' % (np.mean([self.sx_star_laz, self.sy_star_laz])))

    def plot(self, argument_dict=None):
        """Make the astrometric orbit plots.

        Parameters
        ----------
        save_plot
        plot_dir
        name_seed
        ppm_description
        omc2D
        arrow_offset_x
        arrow_offset_y
        arrow_length_factor
        m1_MS
        horizons_file_seed
        orbit_only_panel
        frame_residual_panel
        epoch_omc_description
        orbit_description
        omc_panel

        Returns
        -------

        """
        # set defaults
        if argument_dict is not None:
            default_argument_dict = {'arrow_length_factor': 1.,
                            'horizons_file_seed': None,
                            'frame_omc_description': 'default',
                            'orbit_description': 'default',
                            'scan_angle_definition': 'gaia',
                            }

            for key, value in default_argument_dict.items():
                if key not in argument_dict.keys():
                    argument_dict[key] = value

        if argument_dict['ppm_description'] == 'default':
            argument_dict['ppm_description'] = '$\\varpi={:2.3f}$ mas\n$\mu_\\mathrm{{ra^\\star}}={' \
                        ':2.3f}$ mas/yr\n$\mu_\\mathrm{{dec}}={:2.3f}$ mas/yr'.format(
                self.model_parameters[0]['plx_mas'], self.model_parameters[0]['muRA_mas'],
                self.model_parameters[0]['muDE_mas'])

        if argument_dict['epoch_omc_description'] == 'default':
            argument_dict['epoch_omc_description'] = '$N_e={}$, $N_f={}$, $\Delta t={:.0f}$ d\nDOF$_\\mathrm{{eff}}$={}, ' \
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

        for p in range(self.number_of_companions):
            if argument_dict['orbit_description'] == 'default':
                argument_dict['tmp_orbit_description'] = '$P={:2.3f}$ d\n$e={:2.3f}$\n$\\alpha={:2.3f}$ mas\n$i={:2.3f}$ deg\n$M_1={:2.3f}$ Msun\n$M_2={:2.1f}$ Mjup'.format(self.model_parameters[p]['P_day'], self.model_parameters[p]['ecc'], self.model_parameters[p]['a_mas'], self.model_parameters[p]['i_deg'], self.model_parameters[p]['m1_MS'], self.model_parameters[p]['m2_MJ'])
            else:
                argument_dict['tmp_orbit_description'] = argument_dict['orbit_description']

            theta_p = self.model_parameters[p]
            theta_names = theta_p.keys()
            name_seed_2 = argument_dict['name_seed'] + '_companion{:d}'.format(p)

            if 'm2_MS' in theta_names:
                theta_p['m2_MJ'] = theta_p['m2_MS'] * MS_kg / MJ_kg
            if ('plx_abs_mas' in theta_names) & ('plx_corr_mas' in theta_names):
                theta_p['plx_mas'] = theta_p['plx_abs_mas'] + theta_p['plx_corr_mas']

            orb = OrbitSystem(attribute_dict=theta_p)


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

                fig, axes = pl.subplots(n_rows, n_columns, sharex=True, figsize=(9, 9), facecolor='w',
                                        edgecolor='k', squeeze=False)

                self.insert_orbit_timeseries_plot(orb, argument_dict, ax=axes[0][0])
                if self.data_type == '2d':
                    self.insert_orbit_timeseries_plot(orb, argument_dict, direction='y', ax=axes[0][1])
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

                fig.tight_layout(h_pad=0.0)
                pl.show()
                if argument_dict['save_plot']:
                    if argument_dict['frame_residual_panel']:
                        figure_file_name = os.path.join(argument_dict['plot_dir'], 'orbit_time_{}_frameres.pdf'.format(name_seed_2.replace('.', 'p')))
                    else:
                        figure_file_name = os.path.join(argument_dict['plot_dir'],
                                               'orbit_time_{}.pdf'.format(name_seed_2.replace('.', 'p')))
                    plt.savefig(figure_file_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

    def insert_ppm_plot(self, orb, argument_dict):

        t_curve_mjd_2d = np.sort(np.tile(self.t_curve_MJD, 2))
        # tmp = np.arange(1., len(t_curve_mjd_2d) + 1)
        # xi_curve = np.where(
        #     np.remainder(tmp + 1, 2) == 0)  # index of X coordinates (cpsi = 1) psi =  0 deg
        # yi_curve = np.where(
        #     np.remainder(tmp, 2) == 0)  # index of X coordinates (cpsi = 1) psi =  0 deg
        # cpsi_curve = tmp % 2
        # spsi_curve = (tmp + 1) % 2

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

        if ax is None:
            ax = pl.gca()

        if direction=='x':
            ax.plot(self.t_MJD_epoch - orb.Tref_MJD, self.Xmean_orb, 'ko')
            ax.errorbar(self.t_MJD_epoch - orb.Tref_MJD, self.Xmean_orb, yerr=self.errResidualX,
                                fmt='none', ecolor='k')
            if argument_dict['orbit_description'] is not None:
                pl.text(0.01, 0.99, argument_dict['tmp_orbit_description'], horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes)

        if self.data_type == '1d':
            ax.set_ylabel('Offset along scan (mas)')
            # ax.set_title(self.title)

        elif self.data_type == '2d':
            if direction=='x':
                ax.plot(tmpt_day - orb.Tref_MJD, phi1_curve, 'k-')
                ax.set_ylabel('Offset in RA (mas)')
            elif direction=='y':
                axes[0][1].plot(tmpt_day - orb.Tref_MJD, phi2_curve, 'k-')
                axes[0][1].plot(self.t_MJD_epoch - orb.Tref_MJD, Ymean_orb, 'ko')
                axes[0][1].errorbar(self.t_MJD_epoch - orb.Tref_MJD, Ymean_orb, yerr=self.errResidualY,
                                    fmt='none', ecolor='k')
                axes[0][1].set_ylabel('Offset in Dec (mas)')



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
            ax.set_ylabel('O-C (mas)')

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

        # if self.data_type == '1d':
        #     axes[2][0].plot(self.data.epoch_data['MJD'] - offset_MJD, self.residuals, 'ko', mfc='k', ms=4)
        #     axes[2][0].errorbar(self.data.epoch_data['MJD'] - offset_MJD, self.residuals, yerr=self.data.epoch_data['sigma_da_mas'], fmt='none', ecolor='k')
        #     axes[2][0].axhline(y=0, color='0.5', ls='--', zorder=-50)
        #
        # elif self.data_type == '2d':
        #     axes[2][0].plot(self.T['MJD'][self.xi] - offset_MJD, self.residuals[self.xi], 'ko')
        #     axes[2][0].axhline(y=0, color='0.5', ls='--', zorder=-50)
        #     axes[2][1].plot(self.T['MJD'][self.yi] - offset_MJD, self.residuals[self.yi], 'ko')
        #     axes[2][1].axhline(y=0, color='0.5', ls='--', zorder=-50)
        #     axes[-1][1].set_xlabel('MJD - %3.1f' % offset_MJD)
        #     labels = axes[-1][1].get_xticklabels()
        #     plt.setp(labels, rotation=30)
        #
        # if frame_omc_description is not None:
        #     ax = plt.gca()
        #     pl.text(0.01, 0.99, frame_omc_description, horizontalalignment='left',
        #             verticalalignment='top', transform=ax.transAxes)


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

            # ax.plot(self.t_MJD_epoch - orb.Tref_MJD, self.meanResidualX, 'ko')
            # ax.errorbar(self.t_MJD_epoch - orb.Tref_MJD, self.meanResidualX,
            #                     yerr=self.errResidualX, fmt='none', ecolor='k')
            # ax.axhline(y=0, color='0.5', ls='--', zorder=-50)
            # ax.set_ylabel('O-C (mas)')
            if argument_dict['frame_omc_description'] is not None:
                pl.text(0.01, 0.99, argument_dict['frame_omc_description'], horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes)

        elif self.data_type == '2d':

            if direction=='x':
                ax.plot(self.T['MJD'][self.xi] - offset_MJD, self.residuals[self.xi], 'ko')
                ax.axhline(y=0, color='0.5', ls='--', zorder=-50)


            elif direction=='y':
                ax.plot(self.T['MJD'][self.yi] - offset_MJD, self.residuals[self.yi], 'ko')
                ax.axhline(y=0, color='0.5', ls='--', zorder=-50)

            # ax.plot(self.t_MJD_epoch - orb.Tref_MJD, self.meanResidualY, 'ko')
            # ax.errorbar(self.t_MJD_epoch - orb.Tref_MJD, self.meanResidualY,
            #                     yerr=self.errResidualY, fmt='none', ecolor='k')
            # ax.axhline(y=0, color='0.5', ls='--', zorder=-50)
            # ax.set_ylabel('O-C (mas)')

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

        timestamps_1D, cpsi_curve, spsi_curve, xi_curve, yi_curve = get_spsi_cpsi_for_2Dastrometry(self.t_curve_MJD, scan_angle_definition=argument_dict['scan_angle_definition'])
        orbit_curve = orb.pjGetBarycentricAstrometricOrbitFast(timestamps_1D, spsi_curve, cpsi_curve)
        phi1_curve = orbit_curve[xi_curve]
        phi2_curve = orbit_curve[yi_curve]

        t_epoch_MJD, cpsi_epoch, spsi_epoch, xi_epoch, yi_epoch = get_spsi_cpsi_for_2Dastrometry(self.t_MJD_epoch, scan_angle_definition=argument_dict['scan_angle_definition'])
        orbit_epoch = orb.pjGetBarycentricAstrometricOrbitFast(t_epoch_MJD, spsi_epoch, cpsi_epoch)
        phi1_model_epoch = orbit_epoch[xi_epoch]
        phi2_model_epoch = orbit_epoch[yi_epoch]

        t_frame_mjd, cpsi_frame, spsi_frame, xi_frame, yi_frame = get_spsi_cpsi_for_2Dastrometry(self.data.epoch_data['MJD'], scan_angle_definition=argument_dict['scan_angle_definition'])
        orbit_frame = orb.pjGetBarycentricAstrometricOrbitFast(t_frame_mjd, spsi_frame, cpsi_frame)
        phi1_model_frame = orbit_frame[xi_frame]
        phi2_model_frame = orbit_frame[yi_frame]

        # show periastron
        if 1:

            t_periastron_mjd, cpsi_periastron, spsi_periastron, xi_periastron, yi_periastron = get_spsi_cpsi_for_2Dastrometry(orb.Tp_day, scan_angle_definition=argument_dict['scan_angle_definition'])
            orbit_periastron = orb.pjGetBarycentricAstrometricOrbitFast(t_periastron_mjd, spsi_periastron, cpsi_periastron)
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

                # pass
                #  from yorick code
                #     // psi is the scan angle from north to east (better, from west to north)
                # // scanning direction
                # dx1_mas = cpsi_obs *  myresidual;//*hd.SRES;
                # dy1_mas = spsi_obs *  myresidual;// *hd.SRES;

        elif self.data_type == '2d':
            pl.plot(Xmean_orb, Ymean_orb, 'ko', ms=8)
            pl.errorbar(Xmean_orb, Ymean_orb, xerr=self.errResidualX, yerr=self.errResidualY,
                        fmt='none', ecolor='0.6', zorder=-49)
            for j in range(len(phi1_model_epoch)):
                pl.plot([Xmean_orb[j], phi1_model_epoch[j]], [Ymean_orb[j], phi2_model_epoch[j]],
                        'k--', color='0.7', zorder=-50)

        # show origin
        pl.plot(0, 0, 'kx')

        if argument_dict['tmp_orbit_description'] is not None:
            pl.text(0.01, 0.99, argument_dict['tmp_orbit_description'], horizontalalignment='left',
                    verticalalignment='top', transform=pl.gca().transAxes)


class pDetLim(object):
    """
    Class to support determination of planet detectin limits from astrometry

    """

    def __init__(self, dwNr, M1_Msun, absPlx_mas, M2_jup_grid_N, M2_Mjup_lowlim, M2_Mjup_upplim, N_sim_perPeriod,
                 P_day_grid_N, P_day_grid_min, P_day_grid_max, dwDir, overwrite=0):
        # self.name = name
        self.M1_Msun = M1_Msun
        self.absPlx_mas = absPlx_mas
        self.M2_jup_grid_N = M2_jup_grid_N
        self.N_sim_perPeriod = N_sim_perPeriod
        self.P_day_grid_N = P_day_grid_N
        self.M2_Mjup_lowlim = M2_Mjup_lowlim
        self.M2_Mjup_upplim = M2_Mjup_upplim
        self.P_day_grid_min = P_day_grid_min
        self.P_day_grid_max = P_day_grid_max
        self.dwDir = dwDir
        self.overwrite = overwrite
        self.dwNr = dwNr

        self.N_sim = P_day_grid_N * N_sim_perPeriod * M2_jup_grid_N;  # number of planetary systems generated

        print('Simulations: total number %d (%d periods, %d M2, %d random) ' % (
        self.N_sim, P_day_grid_N, M2_jup_grid_N, N_sim_perPeriod))
        print('Simulations: M2 resolution %3.3f Mjup ' % ((M2_Mjup_upplim - M2_Mjup_lowlim) / self.M2_jup_grid_N))

    def prepareReferenceDataset(self, xfP, useMeanEpochs=1, horizonsFileSeed=None):

        self.T0_MJD = xfP.tref_MJD;

        if useMeanEpochs == 1:  # fastSimu works with epoch averages
            orb_mean = OrbitSystem(P_day=1., ecc=0.0, m1_MS=1.0, m2_MJ=0.0, omega_deg=0., OMEGA_deg=0., i_deg=0.,
                                   Tp_day=0, RA_deg=xfP.RA_deg, DE_deg=xfP.DE_deg, plx_mas=self.absPlx_mas, muRA_mas=0,
                                   muDE_mas=0, Tref_MJD=xfP.tref_MJD)
            #             ppm1dMeas_mean_mas = orb_mean.getPpm(xfP.t_MJD_epoch, horizonsFileSeed=horizonsFileSeed)
            ppm1dMeas_mean_mas = orb_mean.ppm(xfP.t_MJD_epoch, horizons_file_seed=horizonsFileSeed,
                                              psi_deg=xfP.psi_deg)
            C_mean = orb_mean.coeffMatrix
            TableC1_mean = Table(C_mean.T, names=('cpsi', 'spsi', 'ppfact', 'tcpsi', 'tspsi'))
            tmp_mean, xi_mean, yi_mean = xfGetMeanParMatrix(xfP)
            S_mean = np.mat(np.diag(1. / np.power(tmp_mean['sigma_da_mas'], 2)));
            M_mean = np.mat(tmp_mean['da_mas'])
            # res_mean = linfit(M_mean, S_mean, C_mean)
            res_mean = linearfit.LinearFit(M_mean, S_mean, C_mean)
            res_mean.fit()
            # res_mean.makeReadableNumbers()

            self.TableC1_mean = TableC1_mean
            self.tmp_mean = tmp_mean
            self.res_mean = res_mean
            self.S_mean = S_mean
            self.C_mean = C_mean
            # res_mean.disp()

    def run_simulation(self, simuRun=1, log_P_day_grid=True):
        self.M2_jup_grid = np.linspace(self.M2_Mjup_lowlim, self.M2_Mjup_upplim, self.M2_jup_grid_N)
        if log_P_day_grid:
            self.P_day_grid = np.logspace(np.log10(self.P_day_grid_min), np.log10(self.P_day_grid_max),
                                          self.P_day_grid_N)
        else:
            self.P_day_grid = np.linspace(self.P_day_grid_min, self.P_day_grid_max, self.P_day_grid_N)
            # P_day_grid  = np.linspace((P_day_grid_min),(P_day_grid_max),num=P_day_grid_N) # for comparison with GAIA

        #         simuRun = 1;
        simuDir = self.dwDir + 'simu/simuRun%d/' % simuRun;
        if not os.path.exists(simuDir):
            os.makedirs(simuDir)

        mcFileName = simuDir + 'dw%02d_detectionLimits_%d%s.pkl' % (
        self.dwNr, self.N_sim, ('_MA%1.3f' % self.M1_Msun).replace('.', 'p'))
        # meanResiduals = np.zeros((self.N_sim, len(self.res_mean.omc[0])))
        meanResiduals = np.zeros((self.N_sim, len(self.res_mean.residuals)))
        meanResidualRMS = np.zeros(self.N_sim)

        if ((not os.path.isfile(mcFileName)) or (self.overwrite == 1)):

            OMEGA_deg_vals = np.linspace(0, 359, 360)
            simu_OMEGA_deg = np.random.choice(OMEGA_deg_vals, self.N_sim)

            i_deg_vals = np.linspace(0, 179, 180)
            PDF_i_deg = 1. / 2 * np.sin(np.deg2rad(i_deg_vals))
            PDF_i_deg_normed = PDF_i_deg / np.sum(PDF_i_deg)
            simu_i_deg = np.random.choice(i_deg_vals, self.N_sim, p=PDF_i_deg_normed)

            simu_M2_jup = np.zeros(self.N_sim)
            temp_M2 = np.zeros(self.M2_jup_grid_N * self.N_sim_perPeriod)
            for jj in range(self.M2_jup_grid_N):
                tempIdx = np.arange(jj * self.N_sim_perPeriod, (jj + 1) * self.N_sim_perPeriod)
                temp_M2[tempIdx] = self.M2_jup_grid[jj] * np.ones(self.N_sim_perPeriod)

            simu_P_day = np.zeros(self.N_sim)
            for jj in range(self.P_day_grid_N):
                tempIdx = np.arange(jj * self.N_sim_perPeriod * self.M2_jup_grid_N,
                                    (jj + 1) * self.N_sim_perPeriod * self.M2_jup_grid_N)
                simu_P_day[tempIdx] = self.P_day_grid[jj] * np.ones(self.N_sim_perPeriod * self.M2_jup_grid_N)
                simu_M2_jup[tempIdx] = temp_M2;

            simu_T0_day = self.T0_MJD + np.random.rand(self.N_sim) * simu_P_day;

            ecc = 0.;
            omega_deg = 0.;

            if 0 == 1:
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
            spsi = np.array(self.TableC1_mean['spsi'])
            cpsi = np.array(self.TableC1_mean['cpsi'])
            ref_da_mas = np.array(self.tmp_mean['da_mas'])
            # ref_omc_mas = self.res_mean.omc[0]
            ref_omc_mas = self.res_mean.residuals
            for j in range(self.N_sim):
                tot_da_mas = [];
                simu_da_mas = [];
                simu_da_mas = pjGetOrbitFast(P_day=simu_P_day[j], ecc=ecc, m1_MS=self.M1_Msun, m2_MJ=simu_M2_jup[j],
                                             omega_deg=omega_deg, OMEGA_deg=simu_OMEGA_deg[j], i_deg=simu_i_deg[j],
                                             T0_day=simu_T0_day[j], plx_mas=self.absPlx_mas,
                                             t_MJD=np.array(self.tmp_mean['MJD']), spsi=spsi, cpsi=cpsi)
                # orb_simu = OrbitSystem(P_day=simu_P_day[j], ecc=ecc, m1_MS=M1_Msun, m2_MJ = simu_M2_jup[j] , omega_deg=omega_deg, OMEGA_deg=simu_OMEGA_deg[j], i_deg=simu_i_deg[j], Tp_day = simu_T0_day[j], RA_deg=RA_deg,DE_deg=DE_deg,plx_mas = plx_mas, muRA_mas=res.p[3][0],muDE_mas=res.p[4][0] )
                # simu_da_mas = orb_simu.pjGetOrbitFast(0 , t_MJD = tmp_mean['MJD'], psi_deg = psi_deg )#, verbose=0):

                tot_da_mas = ref_da_mas - ref_omc_mas + simu_da_mas  # remove noise structure
                # simu_res = linfit(np.mat(tot_da_mas), self.S_mean, self.C_mean)
                simu_res = linearfit.LinearFit(np.mat(tot_da_mas), self.S_mean, self.C_mean)
                simu_res.fit()
                # meanResiduals[j,:] = np.array(simu_res.omc)[0]
                meanResidualRMS[j] = np.std(np.array(simu_res.residuals))
                if np.mod(j, 10000) == 0:
                    print('\b\b\b\b\b\b\b%07d' % j)
                    # print '\x1b[%07d\r' % j,
            pickle.dump((meanResidualRMS), open(mcFileName, "wb"))

        else:
            meanResidualRMS = pickle.load(open(mcFileName, "rb"))

        self.meanResidualRMS = meanResidualRMS

    def run_simulation_parallel(self, simulation_run_number=1, log_P_day_grid=True, parallel=True):
        '''
        parallelized running of simulations, looping through simulated pseudo-orbits

        :param simulation_run_number:
        :param log_P_day_grid:
        :param parallel:
        :return:
        '''

        # directory to write to
        simulation_dir = os.path.join(self.dwDir, 'simulation/simulation_run_number%d/' % simulation_run_number)
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        # generate grid of companion masses
        self.M2_jup_grid = np.linspace(self.M2_Mjup_lowlim, self.M2_Mjup_upplim, self.M2_jup_grid_N)

        # generate grid of orbital periods (log or linear spacing)
        if log_P_day_grid:
            self.P_day_grid = np.logspace(np.log10(self.P_day_grid_min), np.log10(self.P_day_grid_max),
                                          self.P_day_grid_N)
        else:
            self.P_day_grid = np.linspace(self.P_day_grid_min, self.P_day_grid_max, self.P_day_grid_N)

        # pickle file to save results
        mc_file_name = os.path.join(simulation_dir, 'dw%02d_detectionLimits_%d%s.pkl' % (
            self.dwNr, self.N_sim, ('_MA%1.3f' % self.M1_Msun).replace('.', 'p')))

        # meanResiduals = np.zeros((self.N_sim, len(self.res_mean.omc[0])))
        mean_residual_rms = np.zeros(self.N_sim)

        N_sim_within_loop = self.N_sim_perPeriod * self.M2_jup_grid_N
        # array to hold results, sliced by orbital period
        mean_residual_rms = np.zeros((self.P_day_grid_N, N_sim_within_loop))

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

        if ((not os.path.isfile(mc_file_name)) or (self.overwrite == 1)):
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
            # temp_M2 = np.zeros(self.M2_jup_grid_N * self.N_sim_perPeriod)
            for jj in range(self.M2_jup_grid_N):
                tempIdx = np.arange(jj * self.N_sim_perPeriod, (jj + 1) * self.N_sim_perPeriod)
                simu_M2_jup[tempIdx] = self.M2_jup_grid[jj] * np.ones(self.N_sim_perPeriod)

            # simu_P_day = np.zeros(self.N_sim)
            # for jj in range(self.P_day_grid_N):
            #     tempIdx = np.arange(jj * self.N_sim_perPeriod * self.M2_jup_grid_N,
            #                         (jj + 1) * self.N_sim_perPeriod * self.M2_jup_grid_N)
            #     simu_P_day[tempIdx] = self.P_day_grid[jj] * np.ones(self.N_sim_perPeriod * self.M2_jup_grid_N)
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
            # for line, val in enumerate(list_start_vals):
            #     result = pool.apply_async(fill_array, [val])
            #     array_2D[line, :] = result.get()

            arg_list = []
            for jj, P_day in enumerate(self.P_day_grid):
                # print('Processing period number %d'%jj)

                np.random.seed(random_seed)
                simu_T0_day = self.T0_MJD + np.random.rand(N_sim_within_loop) * P_day

                arg = [P_day, ecc, self.M1_Msun, simu_M2_jup,
                       omega_deg, simu_OMEGA_deg, simu_i_deg,
                       simu_T0_day, self.absPlx_mas,
                       np.array(self.tmp_mean['MJD']), spsi, cpsi, ref_da_mas, ref_omc_mas]
                arg_list.append(arg)


                # result = pool.map(return_residual_rms_array, )
                # mean_residual_rms[jj,:] = result

            import time

            t0 = time.time()

            mean_residual_rms = np.array(pool.map(return_residual_rms_array, arg_list))
            t1 = time.time()
            print('multiprocessing using %d processes finished in %3.3f sec' % (n_processes, t1 - t0))

            pool.close()

            pickle.dump((mean_residual_rms.flatten()), open(mc_file_name, "wb"))

        else:
            mean_residual_rms = pickle.load(open(mc_file_name, "rb"))

        self.meanResidualRMS = mean_residual_rms.flatten()


        # def fill_array(start_val):
        #     return range(start_val, start_val + 10)
        #
        # if 1:
        #     pool = Pool()
        #     list_start_vals = range(40, 60)
        #     array_2D = np.zeros((20, 10))
        #     for line, val in enumerate(list_start_vals):
        #         result = pool.apply_async(fill_array, [val])
        #         array_2D[line, :] = result.get()
        #     pool.close()
        #     print(array_2D)

        #     for j in range(self.N_sim):
        #         # tot_da_mas = [];
        #         # simu_da_mas = [];
        #         simu_da_mas = pjGetOrbitFast(P_day=simu_P_day[j], ecc=ecc, m1_MS=self.M1_Msun, m2_MJ=simu_M2_jup[j],
        #                                      omega_deg=omega_deg, OMEGA_deg=simu_OMEGA_deg[j], i_deg=simu_i_deg[j],
        #                                      Tp_day=simu_T0_day[j], plx_mas=self.absPlx_mas,
        #                                      t_MJD=np.array(self.tmp_mean['MJD']), spsi=spsi, cpsi=cpsi)
        #         # orb_simu = OrbitSystem(P_day=simu_P_day[j], ecc=ecc, m1_MS=M1_Msun, m2_MJ = simu_M2_jup[j] , omega_deg=omega_deg, OMEGA_deg=simu_OMEGA_deg[j], i_deg=simu_i_deg[j], Tp_day = simu_T0_day[j], RA_deg=RA_deg,DE_deg=DE_deg,plx_mas = plx_mas, muRA_mas=res.p[3][0],muDE_mas=res.p[4][0] )
        #         # simu_da_mas = orb_simu.pjGetOrbitFast(0 , t_MJD = tmp_mean['MJD'], psi_deg = psi_deg )#, verbose=0):
        #
        #         tot_da_mas = ref_da_mas - ref_omc_mas + simu_da_mas;  # remove noise structure
        #         simu_res = linfit(np.mat(tot_da_mas), self.S_mean, self.C_mean)
        #         # meanResiduals[j,:] = np.array(simu_res.omc)[0]
        #         mean_residual_rms[j] = np.std(np.array(simu_res.omc)[0])
        #         if np.mod(j, 10000) == 0:
        #             print('\b\b\b\b\b\b\b%07d' % j)
        #             # print '\x1b[%07d\r' % j,
        #     pickle.dump((mean_residual_rms), open(mc_file_name, "wb"))
        #
        # else:
        #     mean_residual_rms = pickle.load(open(mc_file_name, "rb"))
        #
        # self.meanResidualRMS = mean_residual_rms

    def plotSimuResults(self, xfP, factor=1., visplot=1, confidence_limit=0.997, x_axis_unit='day', semilogx=True, y_data_divisor=None, y_data_factor=1., new_figure=True, line_width=2.):

        if xfP.psi_deg is None:
            criterion = np.std([xfP.meanResidualX, xfP.meanResidualY]) * factor;
        else:
            criterion = np.std([xfP.meanResidualX]) * factor;
        print('Detection criterion is %3.3f mas ' % (criterion))
        print('Using confidence limit of {:.3f}'.format(confidence_limit))

        Nsmaller = np.zeros((self.P_day_grid_N, self.M2_jup_grid_N))
        for jj in range(self.P_day_grid_N):
            tempIdx = np.arange(jj * self.N_sim_perPeriod * self.M2_jup_grid_N,
                                (jj + 1) * self.N_sim_perPeriod * self.M2_jup_grid_N)
            for kk in range(self.M2_jup_grid_N):
                pix = np.arange(kk * self.N_sim_perPeriod, (kk + 1) * self.N_sim_perPeriod)
                # Nsmaller[jj,kk] = np.sum( np.std(meanResiduals[tempIdx[pix]],axis=1) <= criterion )
                Nsmaller[jj, kk] = np.sum(self.meanResidualRMS[tempIdx[pix]] <= criterion)

        detLimit = np.zeros((self.P_day_grid_N, 2))
        for jj in range(self.P_day_grid_N):
            try:
                I = np.where(Nsmaller[jj, :] < self.N_sim_perPeriod * (1 - confidence_limit))[0][0]
                try:
                    M2_val = self.M2_jup_grid[I]
                except ValueError:
                    M2_val = np.max(self.M2_jup_grid)
            except IndexError:
                #                 pdb.set_trace()
                M2_val = np.max(self.M2_jup_grid)

            detLimit[jj, :] = [self.P_day_grid[jj], M2_val]

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
                    pl.semilogx(detLimit[:, 0] * x_axis_factor, y_data_divisor/detLimit[:, 1]*y_data_factor, 'k-', lw=line_width)
                else:
                    pl.semilogx(detLimit[:, 0] * x_axis_factor, detLimit[:, 1]*y_data_factor, 'k-', lw=line_width)
            else:
                if y_data_divisor is not None:
                    pl.plot(detLimit[:, 0] * x_axis_factor, y_data_divisor/detLimit[:, 1]*y_data_factor, 'k-', lw=line_width)
                else:
                    pl.plot(detLimit[:, 0] * x_axis_factor, detLimit[:, 1] * y_data_factor, 'k-',
                            lw=line_width)
            pl.title('{:.1f}% confidence limit'.format(confidence_limit * 100))
            if y_data_divisor is not None:
                pl.ylim((0, y_data_divisor/np.max(self.M2_jup_grid)*y_data_factor))
            else:
                pl.ylim((0, np.max(self.M2_jup_grid)*y_data_factor))
            pl.xlabel(x_axis_label)
            if new_figure:
                pl.show()

        self.detLimit = detLimit


def xfGetMeanParMatrix(xfP):
    """ Copied over from functions_and_classes, 2018-08-14, should be deleted

    Parameters
    ----------
    xfP

    Returns
    -------

    """
    if xfP.psi_deg is None:
        cat = Table()
        cat['MJD'] = xfP.t_MJD_epoch
        cat['RA*_mas'] = xfP.Xmean
        cat['DE_mas'] = xfP.Ymean
        cat['sRA*_mas'] = xfP.errResidualX
        cat['sDE_mas'] = xfP.errResidualY
        cat['OB'] = xfP.medi

        # tmp = cat['MJD','fx[1]','fx[2]','fy[1]','fy[2]','RA*_mas','DE_mas','sRA*_mas','sDE_mas','OB','frame'];
        tmp = cat.copy()
        tmp = tablevstack((tmp, tmp))
        tmp.sort('MJD')
        tmp.add_column(Column(name='da_mas', data=np.zeros(len(tmp))))
        tmp.add_column(Column(name='sigma_da_mas', data=np.zeros(len(tmp))))

        spsi = (np.arange(1, len(tmp) + 1) + 1) % 2;  # % first X then Y
        cpsi = (np.arange(1, len(tmp) + 1)) % 2;
        # xi = spsi==0;    #index of X coordinates (cpsi = 1) psi =  0 deg
        # yi = cpsi==0;    #index of Y coordinates (spsi = 1) psi = 90 deg

        xi = np.where(spsi == 0)[0];  # index of X coordinates (cpsi = 1) psi =  0 deg
        yi = np.where(cpsi == 0)[0];  # index of Y coordinates (spsi = 1) psi = 90 deg

        tmp['da_mas'][xi] = tmp['RA*_mas'][xi]
        tmp['da_mas'][yi] = tmp['DE_mas'][yi]
        tmp['sigma_da_mas'][xi] = tmp['sRA*_mas'][xi]
        tmp['sigma_da_mas'][yi] = tmp['sDE_mas'][yi]

    else:
        tmp = Table()
        tmp['MJD'] = xfP.t_MJD_epoch
        tmp['da_mas'] = xfP.Xmean
        tmp['sigma_da_mas'] = xfP.errResidualX
        tmp['OB'] = xfP.medi
        xi = None
        yi = None

    return tmp, xi, yi


def get_spsi_cpsi_for_2Dastrometry( timestamps_2D , scan_angle_definition='hipparcos'):
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
    timestamps_1D = np.sort(np.hstack((timestamps_2D, timestamps_2D)))
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
        yi = np.where(spsi==0)[0]    #index of X coordinates (cpsi = 1) psi =  0 deg
        xi = np.where(cpsi==0)[0]    #index of Y coordinates (spsi = 1) psi = 90 deg

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

    return a_m


# def mass_from_mass_term(mass_term, type='relative'):
#     """Returm mass corresponding to mass term
#
#     Parameters
#     ----------
#     mass_term
#     type
#
#     Returns
#     -------
#
#     """


# def keplerian_secondary_mass(m1_kg, a_m, P_day):
#     """Return companion mass in kg.
#
#     Parameters
#     ----------
#     m1_kg : float
#         primary mass in kg
#     a_m : float
#         barycentric semimajor axis in meter
#     P_day : float
#         orbital period in days
#
#     Returns
#     -------
#
#     """
#     if 0:
#         import sympy as sp
#         c = sp.Symbol('c')
#         g = sp.Symbol('g')
#         m1 = sp.Symbol('m1')
#         m2 = sp.Symbol('m2')
#         # zero_equation = m2 ** 3 / (m1 + m2) ** 2 - c  # == 0
#         zero_equation = g * m2**3 / (m1 + m2)** 2 - c  # == 0
#         res = sp.solvers.solve((zero_equation), (m2))
#         print(res[0])
#         1/0
#
#
#     if 0:
#
#         m1 = m1_kg
#         c = (4. * np.pi**2. * a_m**3. / ((P_day * day2sec)**2. * Ggrav))
#
#         print(m1)
#         print(c)
#         term_1 = (c**2. + 6.*c*m1)
#         term_4 = np.sqrt(-4.*(c**2 + 6*c*m1)**3 + (-2*c**3 - 18. * c**2 * m1 - 27.*c * m1**2)**2)
#         term_2 = (3.*(-c**3 - 9. * c**2 * m1 - 27. * c * m1**2 /2. + term_4/2.)**(1./3.))
#         term_5 = np.sqrt(-4.*(c**2.+ 6*c*m1)**3 + (-2*c**3 - 18. * c**2 * m1 - 27.*c * m1**2)**2)
#         term_3 =     (-c**3 - 9. * c**2 * m1 - 27. * c * m1**2 /2. + term_5/2.)**(1./3.)/3.
#
#         print(term_1, term_2, term_3, term_4, term_5)
#         m2_kg = c/3. - term_1/term_2 - term_3
#
#     else:
#         m1 = m1_kg
#         c = np.abs(4. * np.pi**2. * a_m**3. / ((P_day * day2sec)**2.))
#         g = Ggrav
#         print(c)
#
#         m2_kg = c/(3*g) - (c**2/g**2 + 6*c*m1/g)/(3*(-c**3/g**3 - 9*c**2*m1/g**2 - 27*c*m1**2/(2*g) + np.sqrt(-4*(c**2/g**2 + 6*c*m1/g)**3 + (-2*c**3/g**3 - 18*c**2*m1/g**2 - 27*c*m1**2/g)**2)/2.)**(1./3.)) - (-c**3/g**3 - 9*c**2*m1/g**2 - 27*c*m1**2/(2*g) + np.sqrt(-4*(c**2/g**2 + 6*c*m1/g)**3 + (-2*c**3/g**3 - 18*c**2*m1/g**2 - 27*c*m1**2/g)**2)/2.)**(1./3.)/3.
#
#     return m2_kg


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
    

def pjGet_a_barycentre(m1_MS, m2_MJ, P_day, plx_mas ):
    M = Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
    a_m = ( M / (4. * np.pi**2.) * (P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
    d_pc  = 1./ (plx_mas/1000.)
    a_rad = np.arctan2(a_m,d_pc*pc_m)
    a_mas = a_rad * rad2mas # semimajor axis in mas         
    return a_mas


def pjGet_a_m_barycentre(m1_MS, m2_MJ, P_day ):
    M = Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
    a_m = ( M / (4. * np.pi**2.) * (P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
    return a_m


def pjGet_a_relative(m1_MS, m2_MJ, P_day, plx_mas ):
    a_rel_m = (Ggrav*(m1_MS*MS_kg+m2_MJ*MJ_kg) / 4. /(np.pi**2.) *(P_day*day2sec)**2.)**(1./3.)
#     M = Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
#     a_m = ( M / (4. * np.pi**2.) * (P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
    d_pc  = 1./ (plx_mas/1000.)    
    a_rel_rad = np.arctan2(a_rel_m,d_pc*pc_m)
    a_rel_mas = a_rel_rad * rad2mas # semimajor axis in mas         
    return a_rel_mas


def pjGet_DetectionLimits( m1_MS, Period_day, d_pc, a1_detection_mas ):
    
    a_m = a1_detection_mas / 1.e3 * d_pc * AU_m
    m1_kg = m1_MS * MS_kg    
    P_day = Period_day
    
    m2_kg = pjGet_m2( m1_kg, a_m, P_day )
    m2_MJ = m2_kg / MJ_kg
    return m2_MJ
    
    # p1 =     (4. * np.pi**2.)
    # p2 = (P_day*day2sec)**2
    # p = p2/p1*Ggrav
    # a = (a1_detection_mas / 1.e3 * d_pc * AU_m)**3
    # m1 = m1_MS
    
    # m2 = a/(3*p) + (-a**2/(9*p**2) - 2*a*m1/(3*p))/(-a**3/(27*p**3) - a**2*m1/(3*p**2) - a*m1**2/(2*p) + ((-a**2/(9*p**2) - 2*a*m1/(3*p))**3 + (-2*a**3/(27*p**3) - 2*a**2*m1/(3*p**2) - a*m1**2/p)**2/4)**(1/2))**(1/3) - (-a**3/(27*p**3) - a**2*m1/(3*p**2) - a*m1**2/(2*p) + ((-a**2/(9*p**2) - 2*a*m1/(3*p))**3 + (-2*a**3/(27*p**3) - 2*a**2*m1/(3*p**2) - a*m1**2/p)**2/4)**(1/2))**(1/3)
    
    # # a = (a1_detection_mas/1.e3 * AU_m * d_pc)**3 * (4. * np.pi**2.)
    # # P_day = Period_day
    # # b = a * (P_day*day2sec)**2 / Ggrav
    # # m1 = m1_MS*MS_kg
    # # # m2 = m2_MJ * MJ_kg
    # # m2 = b/3 + (-b**2/9 - 2*b*m1/3)/(-b**3/27 - b**2*m1/3 - b*m1**2/2 + ((-b**2/9 - 2*b*m1/3)**3 + (-2*b**3/27 - 2*b**2*m1/3 - b*m1**2)**2/4)**(1/2))**(1/3) - (-b**3/27 - b**2*m1/3 - b*m1**2/2 + ((-b**2/9 - 2*b*m1/3)**3 + (-2*b**3/27 - 2*b**2*m1/3 - b*m1**2)**2/4)**(1/2))**(1/3)

    # res = m2
    # return res
    
#             #
   
#         # M = Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
#         # a1_detection_mas = ( M / (4. * np.pi**2.) * (P_day*day2sec)**2. )**(1./3.) / AU_m /d_pc*1.e3 

#         from sympy import Eq, Symbol, solve
#         m2_MJ = Symbol('m2_MJ')
#         (a1_detection_mas/1.e3 * AU_m * d_pc)**3 * (4. * np.pi**2.) * (P_day*day2sec)**2. =  Ggrav * (m2_MJ * MJ_kg)**3. / ( m1_MS*MS_kg + m2_MJ*MJ_kg )**2. 

                                        
            
            
        # M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        # a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.) 
        # a_AU = a_m / AU_m
        # a1_mas = a_AU/d_pc*1.e3 
        #


def mean_anomaly(t_day, T0_day, P_day):
    return 2*np.pi*(t_day-T0_day)/P_day        
    # M_rad= 2*pi*(nu_daym1*t_day - phi0)
            
        
def eccentric_anomaly(ecc,t_day, T0_day, P_day):
    # following /Users/sahlmann/astro/palta/processing/src/MIKS-GA4FORS_v0.4/genetic/kepler-genetic.i
    
    M_rad = mean_anomaly(t_day, T0_day, P_day)
    if np.all(ecc) == 0.0:
#         print 'eccentric_anomaly: Assuming circular orbit'
        return M_rad
    else:
        E_rad = np.zeros(len(M_rad))

        E0_rad = M_rad + ecc*np.sin(M_rad)*(1+ecc*np.cos(M_rad)) #valeur initiale
        Enew_rad=E0_rad # initialissation a l'anomalie moyenne
        cnt=0  #compteur d'iterations
        E_rad_tmp = 1000.
        while ( (np.max(np.abs(Enew_rad-E_rad_tmp)) >1.e-8) & (cnt<200) ):
            E_rad_tmp = Enew_rad
            f   =  E_rad_tmp - ecc*np.sin(E_rad_tmp) - M_rad   
            fp  =  1-ecc*np.cos(E_rad_tmp)#derivee de f par rapport a E
            fpp =  ecc*np.sin(E_rad_tmp)
            # //Enew_rad = E_rad_tmp - f/fp //
            # //Enew_rad = E_rad_tmp -2*fp/fpp - sqrt( (fp/fpp)^2 +f) bof
            Enew_rad = E_rad_tmp - 2*fp*f/(2*fp**2-f*fpp) #np.marche tres bien
            cnt+=1
        E_rad=E_rad_tmp
        return E_rad


def RadialVelocitiesConstants(k1_mps,om_rad,ecc):
    
    alpha_mps = +k1_mps*np.cos(om_rad)
    beta_mps  = -k1_mps*np.sin(om_rad)
    delta_mps = +k1_mps*ecc*np.cos(om_rad)
  
    return np.array([alpha_mps,beta_mps,delta_mps])


def TrueAnomaly(ecc,E_rad):
  # if ( ecc > 0.99) {ecc=0.99}//write,"error, ecc>=0.95, set it arbitrary to 0.3"}
  # if ( ecc <  0.00) {ecc=0.00}//write,"error, ecc<0., set it arbitrary to 0.3"}   
  #  if (ecc!=0.) theta_rad = 2.*atan( sqrt((1.+ecc)/(1.-ecc))*tan(E_rad/2.) )
  #  if (ecc==0.) theta_rad = E_rad

    theta_rad = 2.*np.arctan( np.sqrt((1.+ecc)/(1.-ecc))*np.tan(E_rad/2.) )
    
    # BUG FOUND 2016-02-08, NOT SURE WHERE THIS CAME FROM
    #     theta_rad_tmp = 2.*np.arctan( np.sqrt((1.+ecc)/(1.-ecc))*np.tan(E_rad/2.) )
    #     theta_rad = np.arctan2( np.cos(theta_rad_tmp), np.sin(theta_rad_tmp) )
    
    return theta_rad


def RadialVelocitiesKepler(alpha_mps,beta_mps,delta_mps,theta_rad):    
    Vrad_mps   = alpha_mps * np.cos(theta_rad) + beta_mps * np.sin(theta_rad) + delta_mps
    return Vrad_mps

    
def EllipticalRectangularCoordinates(ecc,E_rad):
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


def get_geomElem(TIC):
    '''
    compute geometrical elements a, omega, OMEGA, i
    from the input of A B F G
    '''

    A = TIC[0]
    B = TIC[1]
    F = TIC[2]
    G = TIC[3]
    p = (A ** 2 + B ** 2 + G ** 2 + F ** 2) / 2;
    q = A * G - B * F;

    a_mas = np.sqrt(p + np.sqrt(p ** 2 - q ** 2))
    # i_rad = math.acos(q/(a_mas**2.))
    # omega_rad = (math.atan2(B-F,A+G)+math.atan2(-B-F,A-G))/2.;
    # OMEGA_rad = (math.atan2(B-F,A+G)-math.atan2(-B-F,A-G))/2.;

    i_rad = np.arccos(q / (a_mas ** 2.))
    omega_rad = (np.arctan2(B - F, A + G) + np.arctan2(-B - F, A - G)) / 2.;
    OMEGA_rad = (np.arctan2(B - F, A + G) - np.arctan2(-B - F, A - G)) / 2.;

    i_deg = i_rad * 360 / 2. / np.pi;
    omega_deg = omega_rad * 360. / (2. * np.pi)
    OMEGA_deg = OMEGA_rad * 360. / (2. * np.    pi)

    GE = [a_mas, omega_deg, OMEGA_deg, i_deg];
    return GE;


def pjGet_TIC(GE):
    # /*  DOCUMENT  xjGet_TIC(GE)
    #     compute A B F G from the input of the geometrical elements a, omega, OMEGA, i
    #     GE = [a_mas, omega_deg, OMEGA_deg, i_deg]
    # extern deg2rad
    # local a_mas, omega_rad, OMEGA_rad, i_rad, A, B, F, G

    a_mas     = GE[0]
    omega_rad = np.deg2rad(GE[1])
    OMEGA_rad = np.deg2rad(GE[2])
    i_rad     = np.deg2rad(GE[3])

    A = a_mas * (np.cos(OMEGA_rad)*np.cos(omega_rad)  - np.sin(OMEGA_rad)*np.sin(omega_rad)*np.cos(i_rad))
    B = a_mas * (np.sin(OMEGA_rad)*np.cos(omega_rad)  + np.cos(OMEGA_rad)*np.sin(omega_rad)*np.cos(i_rad))
    F = a_mas * (-np.cos(OMEGA_rad)*np.sin(omega_rad) - np.sin(OMEGA_rad)*np.cos(omega_rad)*np.cos(i_rad))
    G = a_mas * (-np.sin(OMEGA_rad)*np.sin(omega_rad) + np.cos(OMEGA_rad)*np.cos(omega_rad)*np.cos(i_rad))

    TIC = np.array([A,B,F,G])
    return TIC


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
#       print 'Assuming circular orbit'
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

    # see Equation 8 is Sahlmann+2011
    if scan_angle_definition == 'hipparcos':
        phi = (TIC[0]*spsi + TIC[1]*cpsi)*X + (TIC[2]*spsi + TIC[3]*cpsi)*Y
    elif scan_angle_definition == 'gaia':
        #         A            B                   F           G
        phi = (TIC[0]*cpsi + TIC[1]*spsi)*X + (TIC[2]*cpsi + TIC[3]*spsi)*Y

    return phi


def get_ephemeris(center='g@399', target='0', start_time=None, stop_time=None, step_size='5d', verbose=True, out_dir=None, vector_table_output_type=1, output_units='AU-D', overwrite=False, reference_plane='FRAME'):
    """
    Query the JPL Horizons web interface to return the X,Y,Z position of the target body relative to the center body


    See Horizons_doc.pdf available at https://ssd.jpl.nasa.gov/?horizons#email
    Documentation can also be obtained by sending en email with subject "BATCH-LONG" to horizons@ssd.jpl.nasa.gov


    :param center: string
        Horizons object identifier, default is Earth Center 'g@399'
    :param target: string
        Horizons object identifier, default is Solar System Barycenter '0'
    :param start_time: astropy time instance
    :param stop_time: astropy time instance
    :param step_size: string, default is '1d' for 1 day steps
    :return:


    reference_plane = 'FRAME' is for Earth mean equator and equinox
    """
    global ephemeris_dir

    if start_time is None:
        start_time = Time(1950.0, format='jyear')
    if stop_time is None:
        stop_time = Time(2020.0, format='jyear')

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

        print(url)
        try:
            url_stream = urlopen(url)
        except HTTPError as e:
            print("Unable to open URL:", e)
            sys.exit(1)

        content = url_stream.read()
        url_stream.close()


        # print(content.decode())
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


def getParallaxFactors(RA_deg, DE_deg, t_JD, horizons_file_seed=None, verbose=False, instrument=None, overwrite=False):
    
    ephFactor = -1
    RA_rad = np.deg2rad(RA_deg)
    DE_rad = np.deg2rad(DE_deg)

    if instrument is not None:
        instr = np.unique(instrument)
        Nepoch = len(instrument)
        Xip_val = np.zeros(Nepoch)
        Yip_val = np.zeros(Nepoch)
        Zip_val = np.zeros(Nepoch)
        
        for ins in instr:
            idx = np.where( instrument == ins )[0]                
            if verbose:
                print('Getting Parallax factors for %s using Seed: \t%s' % (ins,ephDict[ins]))
            xyzdata = read_ephemeris(ephDict[ins])
            Xip = interp1d(xyzdata['JD'],xyzdata['X'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
            Yip = interp1d(xyzdata['JD'],xyzdata['Y'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
            Zip = interp1d(xyzdata['JD'],xyzdata['Z'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
#             if ins=='JWST':
#                 pdb.set_trace()
            try:    
                Xip_val[idx] = Xip(t_JD[idx])
                Yip_val[idx] = Yip(t_JD[idx])
                Zip_val[idx] = Zip(t_JD[idx])
            except ValueError:
                print('Error in time interpolation for parallax factors: range %3.1f--%3.2f (%s--%s)\n' % (np.min(t_JD[idx]),np.max(t_JD[idx]), Time(np.min(t_JD[idx]),format='jd',scale='utc').iso, Time(np.max(t_JD[idx]),format='jd',scale='utc').iso )),
                print('Ephemeris file contains data from %s to %s' % (Time(np.min(xyzdata['JD']),format='jd').iso, Time(np.max(xyzdata['JD']),format='jd').iso)) 
                pdb.set_trace()
                1/0   

            parfRA  = ephFactor* ( Xip_val*np.sin(RA_rad) - Yip_val*np.cos(RA_rad) )
            parfDE =  ephFactor*(( Xip_val*np.cos(RA_rad) + Yip_val*np.sin(RA_rad) )*np.sin(DE_rad) - Zip_val*np.cos(DE_rad))

    #     horizons_file_seed = 'horizons_XYZ_2009-2019_EQUATORIAL_Paranal_1h'

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
            parfRA  = ephFactor* ( Xip(t_JD)*np.sin(RA_rad) - Yip(t_JD)*np.cos(RA_rad) )
            parfDE =  ephFactor*(( Xip(t_JD)*np.cos(RA_rad) + Yip(t_JD)*np.sin(RA_rad) )*np.sin(DE_rad) - Zip(t_JD)*np.cos(DE_rad))
        except ValueError:
            raise ValueError('Error in time interpolation for parallax factors: \n'
                             'requested range {:3.1f}--{:3.1f} ({}--{})\n'
                             'available range {:3.1f}--{:3.1f} ({}--{})'.format(np.min(t_JD),np.max(t_JD), Time(np.min(t_JD),format='jd',scale='utc').iso, Time(np.max(t_JD),format='jd',scale='utc').iso, np.min(xyzdata['JDTDB']),np.max(xyzdata['JDTDB']), Time(np.min(xyzdata['JDTDB']),format='jd',scale='utc').iso, Time(np.max(xyzdata['JDTDB']),format='jd',scale='utc').iso
                                                                                ) )

        
        
    return [parfRA,parfDE]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# This example uses subclassing, but there is no reason that the proper function
# couldn't be set up and then use FuncAnimation. The code is long, but not
# really complex. The length is due solely to the fact that there are a total
# of 9 lines that need to be changed for the animation as well as 3 subplots
# that need initial set up.
class SubplotAnimation(animation.TimedAnimation):
    def __init__(self,phi,ppm):
        fig = plt.figure(figsize=(9, 4),facecolor='w', edgecolor='k'); pl.clf()
        ax1 = plt.axes([0.1,0.15,0.35,0.8])
        ax2 = plt.axes([0.55,0.15,0.35,0.8])
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax2 = fig.add_subplot(1, 2, 2)
        # ax3 = fig.add_subplot(2, 2, 4)

        self.t = np.linspace(0, 80, len(phi[0]))
        
        self.x1 = ppm[0]
        self.y1 = ppm[1]
        self.x2 = phi[0]
        self.y2 = phi[1]
        self.z = 10 * self.t

        xlabl = '$\Delta \\alpha^\\star$ (mas)'
        ylabl = '$\Delta \delta$ (mas)'
        ax1.set_xlabel(xlabl)
        ax1.set_ylabel(ylabl)

        self.line1  = Line2D([], [], color='black')
        # self.line1.set_data( self.x1        , self.y1)
        self.line1a = Line2D([], [], color='red', linewidth=2)
        self.line1e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        ax1.add_line(self.line1 )
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)
        ax1.plot(self.x1,self.y1,'k:')
        ax2.plot(0,0,'kx')
        # ax1.axis('equal')
        # ax1.set_xlim(-200, 200)
        # ax1.set_ylim(-200, 200)
        ax1.invert_xaxis()
        ax1.set_aspect('equal', 'datalim')

        # ax1.set_aspect('equal')

        ax2.set_xlabel(xlabl)
        ax2.set_ylabel(ylabl)
        self.line2  = Line2D([], [], color='black')
        self.line2a = Line2D([], [], color='red', linewidth=2)
        self.line2e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        ax2.add_line(self.line2 )
        ax2.add_line(self.line2a)
        ax2.add_line(self.line2e)
        ax2.plot(self.x2,self.y2,'k:')
        ax2.invert_xaxis()
        ax2.set_aspect('equal', 'datalim')
    
        # ax2.set_xlim(-1, 1)
        # ax2.set_ylim(0, 800)

        # ax3.set_xlabel('x')
        # ax3.set_ylabel('z')
        # self.line3 = Line2D([], [], color='black')
        # self.line3a = Line2D([], [], color='red', linewidth=2)
        # self.line3e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        # ax3.add_line(self.line3)
        # ax3.add_line(self.line3a)
        # ax3.add_line(self.line3e)
        # ax3.set_xlim(-1, 1)
        # ax3.set_ylim(0, 800)

        # self._drawn_artists = [self.line1, self.line1a, self.line1e,
        #     self.line2, self.line2a, self.line2e]
        
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True, repeat = False)

    def _draw_frame(self, framedata):
        i = framedata
        # head = i - 1
        head = i
        head_len = 10
        head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

        self.line1.set_data( self.x1[:i]        , self.y1[:i])
        self.line1a.set_data(self.x1[head_slice], self.y1[head_slice])
        self.line1e.set_data(self.x1[head]      , self.y1[head])

        self.line2.set_data( self.x2[:i]        , self.y2[:i])
        self.line2a.set_data(self.x2[head_slice], self.y2[head_slice])
        self.line2e.set_data(self.x2[head]      , self.y2[head])

        # self.line3.set_data(self.x[:i], self.z[:i])
        # self.line3a.set_data(self.x[head_slice], self.z[head_slice])
        # self.line3e.set_data(self.x[head], self.z[head])

        self._drawn_artists = [self.line1, self.line1a, self.line1e,
            self.line2, self.line2a, self.line2e]
            # self.line3, self.line3a, self.line3e]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines =  [self.line1, self.line1a, self.line1e,
            self.line2, self.line2a, self.line2e]
            # self.line3, self.line3a, self.line3e]
        for l in lines:
            l.set_data([], [])


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
    TIC     = pjGet_TIC( [ a_mas   , omega_deg     , OMEGA_deg, i_deg ] ) #Thiele-Innes constants
    phi1 = astrom_signalFast(t_MJD,spsi,cpsi,ecc,P_day,T0_day,TIC) 
    return phi1


def dcr_coefficients(aux):
    """
    DCR parameter computations (following Sahlmann+13)

    :param aux: astropy table, has to contain columns named

    :return:
    """

    # Nmes1D = aux.shape[0] * 2

    temp = aux['temperature'].data # Celcius
    pres = aux['pressure'].data # mbar

    # temp = aux[:, 4]
    # pres = aux[:, 5]
    f3m = (1. - (temp - 11.) / (273. + 11.)) * (1. + (pres - 744.) / 744.)

    # zenith angle
    # z_rad = np.deg2rad(90. - aux[:, 9])
    z_rad = np.deg2rad(90. - aux['tel_altitude'].data)

    lat_rad = np.deg2rad(aux['geo_latitude'].data)
    dec_rad = np.deg2rad(aux['dec'].data)
    azi_rad = np.deg2rad(aux['tel_azimuth'].data)
    # lat_rad = np.deg2rad(aux[:, 10])
    # dec_rad = np.deg2rad(aux[:, 11])
    # azi_rad = np.deg2rad(aux[:, 8])
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
    #     xDCRfactor = xfactor*mat(f1ym).T # WATCH OUT FOR THE X_Y flip here!!!!
    #     yDCRfactor = yfactor*mat(f1xm).T

    return xDCRfactor, yDCRfactor


class ImagingAstrometryData(object):
    """
    structure class for 2D imaging astrometry
    """

    def __init__(self, data_table, out_dir=None):

        # sort data table by increasing time
        self.time_column_name = 'MJD'
        data_table.sort(self.time_column_name)
        
        self.data_table = data_table
        self.number_of_frames = len(np.unique(self.data_table['frame']))
        self.number_of_observing_blocks = len(np.unique(self.data_table['OB']))
        self.observing_time_span_day = np.ptp(data_table[self.time_column_name])
        
        # unique Julian dates of observations, i.e. of 2D astrometry
        self.observing_times_2D_MJD, unique_index = np.unique(np.array(data_table[self.time_column_name]), return_index=True)
        self.data_2D = self.data_table[unique_index]
        
        self.number_of_1D_measurements = 2 * len(self.data_2D)
        
        self.simbad_object_name = None
        if out_dir is not None:
            self.out_dir = out_dir
        else:
            self.out_dir = os.getcwd()   
        
    def info(self):
        print('Number of OBs: \t %d' % self.number_of_observing_blocks)
        print('Number of frames / measurements: \t %d / %d' % (self.number_of_frames,self.number_of_1D_measurements))
        print('Observation time span: \t %3.1f days' % self.observing_time_span_day)
        

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

        # TODO
        # clarify use of tdb here!
        observing_times_2D_TDB_JD = Time(self.observing_times_2D_MJD, format='mjd', scale='utc').tdb.jd
        
        # compute parallax factors, this is a 2xN_obs array
        observing_parallax_factors = getParallaxFactors(self.RA_deg, self.Dec_deg, observing_times_2D_TDB_JD, horizons_file_seed=earth_ephemeris_file_seed, verbose=verbose, overwrite=overwrite)
          
        # set reference epoch for position and computation of proper motion coefficients tspsi and tcpsi   
        if reference_epoch_MJD is None:
            self.reference_epoch_MJD = np.mean(self.observing_times_2D_MJD)
        else:
            self.reference_epoch_MJD = reference_epoch_MJD    

        # time relative to reference epoch in years for proper motion coefficients
        observing_relative_time_2D_year = (self.observing_times_2D_MJD - self.reference_epoch_MJD)/year2day

        observing_relative_time_1D_year, observing_1D_cpsi, observing_1D_spsi, self.observing_1D_xi, self.observing_1D_yi = get_spsi_cpsi_for_2Dastrometry( observing_relative_time_2D_year )

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
            
            
        self.linear_parameter_coefficients_table = tablehstack((self.five_parameter_coefficients_table,self.dcr_parameter_coefficients_table))
        self.linear_parameter_coefficients_array = np.array([self.linear_parameter_coefficients_table[c].data for c in self.linear_parameter_coefficients_table.colnames])
#     tmp['d'][xi]   = xfactor * tmp['fx[2]'][xi]
#     tmp['rho'][yi] = yfactor * tmp['fy[1]'][yi]
#     tmp['d'][yi]   = yfactor * tmp['fy[2]'][yi]
                            
    
    
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


def get_theta_best_genome(best_genome_file, reference_time_MJD, theta_names, m1_MS, instrument=None, verbose=False):
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

    if instrument != 'FORS2':
        best_genome.remove_column('d_mas')

    # if verbose:
    if 0:
        for i in range(len(best_genome)):
            for c in best_genome.colnames:
                print('Planet %d: %s \t %3.3f' % (i+1, c, best_genome[c][i]))

    thiele_innes_constants = np.array([best_genome[c] for c in ['A','B','F','G']])

    a_mas, omega_deg, OMEGA_deg, i_deg = get_geomElem(thiele_innes_constants)
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

        # if i == 0:
        #     theta_best_genome = np.array([best_genome[c][i] for c in col_list])
        # else:
        #     theta_best_genome = np.vstack((theta_best_genome, np.array([best_genome[c][i] for c in col_list])))

    if verbose:
        for i in range(len(best_genome)):
            theta = parameters[i]
            for key,value in theta.items():
                print('Planet %d: Adopted: %s \t %3.3f' % (i, key, value))
 
    # return theta_best_genome
    return parameters
