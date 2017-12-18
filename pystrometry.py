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


def getB(m1,m2):
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
    def __init__(self, P_day=100, ecc=0, m1_MS=1, m2_MJ = 1, omega_deg=0., OMEGA_deg=0., i_deg=90., T0_day = 0., RA_deg=0.,DE_deg=0.,plx_mas = 25., muRA_mas=20.,muDE_mas=50., gamma_ms=0., rvLinearDrift_mspyr = None, rvQuadraticDrift_mspyr = None,rvCubicDrift_mspyr=None ,Tref_MJD=None ): 
        self.P_day = P_day
        self.ecc = ecc
        self.m1_MS= m1_MS
        self.m2_MJ= m2_MJ
        self.omega_deg = omega_deg
        self.OMEGA_deg = OMEGA_deg
        self.i_deg = i_deg
        self.T0_day = T0_day
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

    def pjGetOrbit(self,N,Norbit=None,t_MJD=None, psi_deg=None, verbose=0, returnMeanAnomaly=0, returnTrueAnomaly=0 ):
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
            print("omega = %2.1f deg, OMEGA = %2.1f deg, T0 = %2.1f day " % (self.omega_deg, self.OMEGA_deg,self.T0_day))

        omega_rad = np.deg2rad(self.omega_deg)
        OMEGA_rad = np.deg2rad(self.OMEGA_deg)
        i_rad =     np.deg2rad(self.i_deg)
    
        #*************SIMULATION*PARAMATERS*********************************************
        if Norbit is not None:
            t_day = np.linspace(0,self.P_day*Norbit,N) + self.T0_day
        elif t_MJD is not None:
            t_day = t_MJD
            N = len(t_MJD)

        # R_rv = 50   #number of RV observations
        # R_aric = 50   #number of astrometric observations
        # gnoise_rv = 0.5  # noise level on RV in m/s RMS
        # s_rv = 10.       # error bar on RV measurement in m/s
        # gnoise_aric = 0.05  # noise level on astrometry in mas RMS
        # s_aric = 0.1       # error bar on astromeric measurement in mas 
        # t_day = span(0,P_day*Norbit,N) + T0_day   # time vector  
    
        #**************RADIAL*VELOCITY**************************************************
    
        E_rad = eccentric_anomaly(self.ecc,t_day,self.T0_day,self.P_day) # eccentric anomaly
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
    
        # print('i_deg = %3.2f, om_deg = %3.2f, k1_mps = %3.2f, phi0 = %3.2f, t_day[0] = %3.2f, rv[0] = %3.2f, THETA_rad[0] = %3.2f, E_rad[0] = %2.2f' % (self.i_deg,self.omega_deg,k1, self.T0_day/self.P_day, t_day[0],rv_ms[0], THETA_rad[0],E_rad[0])

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
            phi1 = astrom_signal(t_day,psi_deg,self.ecc,self.P_day,self.T0_day,TIC)
            phi1_rel = astrom_signal(t_day,psi_deg,self.ecc,self.P_day,self.T0_day,TIC_rel)
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
        
            phi1 = astrom_signal(t_day,psi_deg1,self.ecc,self.P_day,self.T0_day,TIC)
            phi2 = astrom_signal(t_day,psi_deg2,self.ecc,self.P_day,self.T0_day,TIC)
            phi1_rel = astrom_signal(t_day,psi_deg1,self.ecc,self.P_day,self.T0_day,TIC_rel)
            phi2_rel = astrom_signal(t_day,psi_deg2,self.ecc,self.P_day,self.T0_day,TIC_rel)
    
        if returnMeanAnomaly:
            M_rad = mean_anomaly(t_day,self.T0_day,self.P_day)
            return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel, M_rad]
            
        elif returnTrueAnomaly:
#           M_rad = mean_anomaly(t_day,self.T0_day,self.P_day)
            return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel, THETA_rad, TIC_rel]
            
        return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel]


    def pjGetRV(self,t_day):
    #    updated: J. Sahlmann   25.01.2016   STScI/ESA
        
        m2_MS = self.m2_MJ * MJ_kg/MS_kg# #companion mass in units of SOLAR mass     
        omega_rad = np.deg2rad(self.omega_deg)
        i_rad =     np.deg2rad(self.i_deg)
    
        #**************RADIAL*VELOCITY**************************************************    
        E_rad = eccentric_anomaly(self.ecc,t_day,self.T0_day,self.P_day) # eccentric anomaly
        M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m
        a_AU = a_m / AU_m #  in AU
        
        # damien's method
        THETA_rad = TrueAnomaly(self.ecc,E_rad)
        k1 = 2. * np.pi * a_m * np.sin(i_rad) / ( self.P_day*day2sec * (1.-self.ecc**2)**(1./2.) ) #RV semiamplitude            
        a_mps = RadialVelocitiesConstants(k1,omega_rad,self.ecc)
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
                                                       
        return rv_ms


    
    def pjGetOrbitFast(self,N,Norbit=None,t_MJD=None, psi_deg=None, verbose=0):
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
        
        phi1 = astrom_signal(t_day,psi_deg,self.ecc,self.P_day,self.T0_day,TIC)
        phi1_rel = np.nan #astrom_signal(t_day,psi_deg,self.ecc,self.P_day,self.T0_day,TIC_rel)
        phi2 = np.nan
        phi2_rel = np.nan
        rv_ms=np.nan
    
        return [phi1 ,phi2, t_day, rv_ms, phi1_rel ,phi2_rel]


    def pjGetBarycentricAstrometricOrbitFast(self,t_MJD, spsi, cpsi):
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
        #       phi1 = astrom_signalFast(t_MJD,spsi,cpsi,ecc,P_day,T0_day,TIC) 
        #       return phi1
               
#         t_MJD = self.MjdUsedInTcspsi
    
        m2_MS = self.m2_MJ * MJ_kg/MS_kg# #companion mass in units of SOLAR mass 
        d_pc  = 1./ (self.plx_mas/1000.)    

        omega_rad = np.deg2rad(self.omega_deg)
        OMEGA_rad = np.deg2rad(self.OMEGA_deg)
        i_rad = np.deg2rad(self.i_deg)
    
        #**************ASTROMETRY********************************************************
        
        M = Ggrav * (self.m2_MJ * MJ_kg)**3. / ( self.m1_MS*MS_kg + self.m2_MJ*MJ_kg )**2. # mass term for the barycentric orbit of the primary mass
        a_m = ( M / (4. * np.pi**2.) * (self.P_day*day2sec)**2. )**(1./3.)  # semimajor axis of the primary mass in m        
        a_rad = np.arctan2(a_m,d_pc*pc_m)
        a_mas = a_rad * rad2mas # semimajor axis in mas         
        TIC     = pjGet_TIC( [ a_mas   , self.omega_deg     , self.OMEGA_deg, self.i_deg ] ) #Thiele-Innes constants
        phi1 = astrom_signalFast(t_MJD,spsi,cpsi,self.ecc,self.P_day,self.T0_day,TIC) 
        return phi1


    def relative_orbit_fast(self,t_MJD, spsi, cpsi, unit = 'mas', shift_omega_by_pi = True, coordinate_system='cartesian'):
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
        TIC     = pjGet_TIC( [ a_rel   ,   omega_rel_deg   , self.OMEGA_deg, self.i_deg ] ) 
        
        # by default these are cartesian coordinates
        phi1 = astrom_signalFast(t_MJD,spsi,cpsi,self.ecc,self.P_day,self.T0_day,TIC) 
        
        # compute polar coordinates if requested
        if coordinate_system=='polar':
        	xi = np.where(cpsi==1)[0]
        	yi = np.where(cpsi==0)[0]
        	rho = np.sqrt(phi1[xi]**2 + phi1[yi]**2)
        	phi_deg = np.rad2deg(np.arctan2(phi1[xi],phi1[yi]))%360.
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
#             print("omega = %2.1f deg, OMEGA = %2.1f deg, T0 = %2.1f day " % (self.omega_deg, self.OMEGA_deg,self.T0_day))
#
#         omega_rad = np.deg2rad(self.omega_deg)
#         OMEGA_rad = np.deg2rad(self.OMEGA_deg)
#         i_rad     = np.deg2rad(self.i_deg)
#
#         #*************SIMULATION*PARAMATERS*********************************************
#         # Norbit = 2.   #number of orbits covered
#         # t_day = np.linspace(0,self.P_day*Norbit,N) + self.T0_day
#
#         # R_rv = 50   #number of RV observations
#         # R_aric = 50   #number of astrometric observations
#         # gnoise_rv = 0.5  # noise level on RV in m/s RMS
#         # s_rv = 10.       # error bar on RV measurement in m/s
#         # gnoise_aric = 0.05  # noise level on astrometry in mas RMS
#         # s_aric = 0.1       # error bar on astromeric measurement in mas
#         # t_day = span(0,P_day*Norbit,N) + T0_day   # time vector
#
#
#         #**************RADIAL*VELOCITY**************************************************
#
#         # E_rad = eccentric_anomaly(self.ecc,t_day,self.T0_day,self.P_day) # eccentric anomaly
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
#         # phi1 = astrom_signal(t_day,psi_deg1,self.ecc,self.P_day,self.T0_day,TIC)
#         # phi2 = astrom_signal(t_day,psi_deg2,self.ecc,self.P_day,self.T0_day,TIC)
    
    

#     def getPpm(self,t_MJD,psi_deg=None, offsetRA_mas=0, offsetDE_mas=0, tref_MJD=None, externalParallaxFactors=None, horizons_file_seed=None):
    def getPpm(self,t_MJD,psi_deg=None, offsetRA_mas=0, offsetDE_mas=0, externalParallaxFactors=None, horizons_file_seed=None, instrument=None, verbose=0):
        
        Nframes = len(t_MJD)
        t_JD = t_MJD + 2400000.5
        if externalParallaxFactors is not None:
            parf = externalParallaxFactors
        else:                
            parf = getParallaxFactors(self.RA_deg,self.DE_deg,t_JD, horizons_file_seed=horizons_file_seed,verbose=verbose,instrument=instrument)
                        
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
            tmp = np.arange(1,2*Nframes+1)
            spsi = (np.arange(1,2*Nframes+1)+1)%2# % first X then Y
            cpsi = (np.arange(1,2*Nframes+1)  )%2
            t = np.sort(np.tile(trel_year ,2))
            
        tspsi = t*spsi
        tcpsi = t*cpsi
        
        if psi_deg is not None:
#             if np.all(psi_deg==0):
#                 ppfact = parf[0]               

#           added JSA 2016-03-07
            ppfact = parf[0] * cpsi + parf[1] * spsi    # see Sahlmann+11 Eq. 1 / 8
                 
        else:
            xi = spsi==0    #index of X coordinates (cpsi = 1) psi =  0 deg
            yi = cpsi==0    #index of Y coordinates (spsi = 1) psi = 90 deg    
            ppfact = np.zeros(2*Nframes)
            ppfact[xi] = parf[0]
            ppfact[yi] = parf[1]
                        
        C = np.array([cpsi,spsi,ppfact,tcpsi,tspsi])
        self.coeffMatrix = C
        self.timeUsedInTcspsi = np.array(t)
        if psi_deg is not None:
            self.MjdUsedInTcspsi = t_MJD
        else:
            self.MjdUsedInTcspsi = np.array(np.sort(np.tile(t_MJD ,2)))

        inVec = np.array([offsetRA_mas,offsetDE_mas,self.plx_mas,self.muRA_mas,self.muDE_mas])
        ppm = np.dot(C.T,inVec)
        if psi_deg is not None:
            return ppm
        else:
            ppm2d = [ppm[xi],ppm[yi]]
            return ppm2d

    def plot_orbits(self, timestamps_curve_2D=None, timestamps_probe_2D=None, timestamps_probe_2D_label=None, delta_mag=None, N_orbit=1., N_curve=100., save_plot=False, plot_dir=None):
        """
        Plot barycentric, photocentric, and relative orbits in two panels
        :param timestamps_curve_2D:
        :param timestamps_probe_2D:
        :param delta_mag:
        :param N_orbit:
        :param N_curve:
        :return:
        """

        if timestamps_curve_2D is None:
            timestamps_curve_2D = np.linspace(self.T0_day,self.T0_day+N_orbit+self.P_day,N_curve)

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
            phi0_probe_photocentre = (f - beta) * self.relative_orbit_fast(timestamps_probe_1D, spsi_probe, cpsi_probe, shift_omega_by_pi = False)


        # barycentric orbit of M1
        phi0_curve_barycentre = self.pjGetBarycentricAstrometricOrbitFast(timestamps_curve_1D, spsi_curve, cpsi_curve)
        phi0_probe_barycentre = self.pjGetBarycentricAstrometricOrbitFast(timestamps_probe_1D, spsi_probe, cpsi_probe)

        n_figure_columns = 2
        n_figure_rows = 1
        # fig, axes = pl.subplots(n_figure_rows, n_figure_columns, figsize=(n_figure_columns*6, n_figure_rows*5), facecolor='w', edgecolor='k', sharex=True, sharey=True)
        fig, axes = pl.subplots(n_figure_rows, n_figure_columns, figsize=(n_figure_columns*6, n_figure_rows*5), facecolor='w', edgecolor='k', sharex=False, sharey=False)
        # plot smooth orbit curve
        axes[0].plot(phi0_curve_barycentre[xi_curve] ,phi0_curve_barycentre[yi_curve],'k--',lw=1, label='Barycentre')
        # plot individual epochs    
        if timestamps_probe_2D is not None:
            axes[0].plot(phi0_probe_barycentre[xi_probe],phi0_probe_barycentre[yi_probe],'bo',mfc='0.7', label=timestamps_probe_2D_label)

        if delta_mag is not None:
            axes[0].plot(phi0_curve_photocentre[xi_curve],phi0_curve_photocentre[yi_curve],'k-',lw=1, label='Photocentre')
            if timestamps_probe_2D is not None:
                axes[0].plot(phi0_probe_photocentre[xi_probe],phi0_probe_photocentre[yi_probe],'bo')

        # plot origin = position of barycentre
        axes[0].plot(0,0,'kx')
        axes[0].axhline(y=0,color='0.7',ls='--',zorder=-50)
        axes[0].axvline(x=0,color='0.7',ls='--',zorder=-50)

        axes[0].set_xlabel('Offset in Right Ascension (mas)')
        axes[0].set_ylabel('Offset in Declination (mas)')
        axes[0].axis('equal')
        axes[0].invert_xaxis()
        axes[0].legend(loc='best')
        axes[0].set_title('Bary-/photocentric orbit of M1')

        # second panel
        # plot smooth orbit curve
        axes[1].plot(phi0_curve_relative[xi_curve],phi0_curve_relative[yi_curve],'k-',lw=1)
        # plot individual epochs
        if timestamps_probe_2D is not None:
            axes[1].plot(phi0_probe_relative[xi_probe],phi0_probe_relative[yi_probe], 'bo', label=timestamps_probe_2D_label)
        # plot origin = position of primary
        axes[1].plot(0,0,'kx')
        axes[1].axhline(y=0,color='0.7',ls='--',zorder=-50)
        axes[1].axvline(x=0,color='0.7',ls='--',zorder=-50)

        axes[1].set_xlabel('Offset in Right Ascension (mas)')
        axes[1].axis('equal')
        axes[1].legend(loc='best')
        axes[1].set_title('Relative orbit of M2 about M1')
        if not axes[1]._sharex:
            axes[1].invert_xaxis()
        pl.show()
        if save_plot:
            figName = os.path.join(plot_dir, 'astrometric_orbits.pdf')
            plt.savefig(figName,transparent=True,bbox_inches='tight',pad_inches=0.05)


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
                 outlier_sigma_threshold=2.):

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

                # if epoch_outlier_dir is not None:
                outliers_x = np.abs(omc[tmpIndexX] - np.mean(omc[tmpIndexX])) > outlier_sigma_threshold * \
                                                                                self.stdResidualX[jj]
                if any(outliers_x):
                    tmp_1D_index_x = np.where(outliers_x)[0]
                    print('Detected %d X-residual outliers (%2.1f sigma) in epoch %d' % (
                    len(tmp_1D_index_x), outlier_sigma_threshold, jj), end='')
                    print(np.abs(omc[tmpIndexX] - np.mean(omc[tmpIndexX]))[tmp_1D_index_x], end='')
                    for ii in tmp_1D_index_x:
                        print('{:.12f}'.format(self.T['MJD'][tmpIndexX[ii]]), end=',')
                    print()

                    outlier_1D_index = np.hstack((outlier_1D_index, tmpIndexX[tmp_1D_index_x]))
                    # outlier_1D_index.append(tmpIndexX[tmp_1D_index_x].tolist())

                outliers_y = np.abs(omc[tmpIndexY] - np.mean(omc[tmpIndexY])) > outlier_sigma_threshold * \
                                                                                self.stdResidualY[jj]
                if any(outliers_y):
                    tmp_1D_index_y = np.where(outliers_y)[0]
                    print('Detected %d Y-residual outliers (%2.1f sigma) in epoch %d' % (
                    len(tmp_1D_index_y), outlier_sigma_threshold, jj), end='')
                    print(np.abs(omc[tmpIndexY] - np.mean(omc[tmpIndexY]))[tmp_1D_index_y], end='')
                    for ii in tmp_1D_index_y:
                        print('{:.12f}'.format(self.T['MJD'][tmpIndexY[ii]]), end=',')
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
                 residual_y_axis_limit=None):

        if self.noParallaxFit != 1:
            orb = OrbitSystem(P_day=1., ecc=0.0, m1_MS=1.0, m2_MJ=0.0, omega_deg=0., OMEGA_deg=0., i_deg=0., T0_day=0,
                              RA_deg=self.RA_deg, DE_deg=self.DE_deg, plx_mas=self.p[2], muRA_mas=self.p[3],
                              muDE_mas=self.p[4], Tref_MJD=self.tref_MJD)
        else:
            orb = OrbitSystem(P_day=1., ecc=0.0, m1_MS=1.0, m2_MJ=0.0, omega_deg=0., OMEGA_deg=0., i_deg=0., T0_day=0,
                              RA_deg=self.RA_deg, DE_deg=self.DE_deg, plx_mas=0, muRA_mas=self.p[2],
                              muDE_mas=self.p[3])

        if separate_residual_panels:
            n_subplots = 3
        else:
            n_subplots = 2



        ##################################################################
        # Figure with on-sky motion only, showing individual frames
        fig = pl.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
        pl.clf()

        if instrument is None:
            if psi_deg is None:
                ppm_curve = orb.getPpm(self.tmodel_MJD, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                                       horizons_file_seed=horizons_file_seed, psi_deg=psi_deg)
            ppm_meas = orb.getPpm(self.t_MJD_epoch, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
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
                ppm_curve = orb.getPpm(self.tmodel_MJD, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                                       horizons_file_seed=horizons_file_seed, psi_deg=psi_deg)
            ppm_meas = orb.getPpm(self.t_MJD_epoch, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
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
                    ppm_curve = orb.getPpm(self.tmodel_MJD, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
                                           instrument=tmpInstrument, psi_deg=psi_deg)
                    pl.plot(ppm_curve[0], ppm_curve[1], c=myColours[jjj], ls='-')
                    pl.plot(self.Xmean[idx], self.Ymean[idx], marker='o', mfc=myColours[jjj], mec=myColours[jjj],
                            ls='None')
            ppm_meas = orb.getPpm(self.t_MJD_epoch, offsetRA_mas=self.p[0], offsetDE_mas=self.p[1],
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
            dt = 0.05
            xt = pl.xlim()[0] + dt * np.diff(pl.xlim())[0]
            yt = pl.ylim()[1] - dt * np.diff(pl.ylim())[0]
            pl.text(xt, yt, descr)

        pl.subplot(n_subplots, 1, 2)
        epochTime = self.t_MJD_epoch - self.tref_MJD
        epochOrdinateLabel = 'MJD - %3.1f' % self.tref_MJD

        pl.plot(epochTime, self.meanResidualX, 'ko', color='0.7', label='RA')
        pl.errorbar(epochTime, self.meanResidualX, yerr=self.errResidualX, fmt=None, ecolor='0.7')
        plt.axhline(y=0, color='0.5', ls='--', zorder=-50)
        pl.ylabel('O-C (mas)')
        if residual_y_axis_limit is not None:
            pl.ylim((-residual_y_axis_limit, residual_y_axis_limit))
        if psi_deg is None:
            if separate_residual_panels:
                pl.subplot(n_subplots, 1, 3)

            pl.plot(epochTime, self.meanResidualY, 'ko', label='Dec')
            pl.errorbar(epochTime, self.meanResidualY, yerr=self.errResidualY, fmt=None, ecolor='k')
            plt.axhline(y=0, color='0.5', ls='--', zorder=-50)
            pl.ylabel('O-C (mas)')
            if residual_y_axis_limit is not None:
                pl.ylim((-residual_y_axis_limit, residual_y_axis_limit))

        if not separate_residual_panels:
            pl.legend(loc='best')

        if instrument is not None:
            for jjj, ins in enumerate(instr):
                idx = np.where(instrument == ins)[0]
                pl.plot(epochTime[idx], self.meanResidualY[idx], marker='o', mfc=myColours[jjj], mec=myColours[jjj],
                        ls='None', label=ins)
            pl.legend(loc='best')

        pl.xlabel(epochOrdinateLabel)
        fig.tight_layout(h_pad=0.0)
        pl.show()
        # pdb.set_trace()

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
                        fmt=None, ecolor='k')
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
            pl.errorbar(epochTime, self.meanResidualX, yerr=self.errResidualX, fmt=None, ecolor='0.7')
            pl.plot(epochTime, self.meanResidualY, 'ko')
            pl.errorbar(epochTime, self.meanResidualY, yerr=self.errResidualY, fmt=None, ecolor='k')
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


def get_spsi_cpsi_for_2Dastrometry( timestamps_2D ):
    '''
    xi = spsi==0    #index of X coordinates (cpsi = 1) psi =  0 deg
    yi = cpsi==0    #index of Y coordinates (spsi = 1) psi = 90 deg    

    '''

    # every 2D timestamp is duplicated to obtain the 1D timestamps 
    timestamps_1D = np.sort(np.hstack((timestamps_2D, timestamps_2D))) 
    
    # compute cos(psi) and sin(psi) factors assuming orthogonal axes    
    spsi = (np.arange(1,len(timestamps_1D)+1)+1)%2# % first X then Y
    cpsi = (np.arange(1,len(timestamps_1D)+1)  )%2

    # indices of X and Y measurements
    xi = np.where(spsi==0)[0]    #index of X coordinates (cpsi = 1) psi =  0 deg
    yi = np.where(cpsi==0)[0]    #index of Y coordinates (spsi = 1) psi = 90 deg    

    return timestamps_1D, cpsi, spsi, xi, yi


def pjGet_m2( m1_kg, a_m, P_day ):
    '''
    # extern day2sec, Gc
    # local a,b,c,m2_kg
    # // return m2 when m1, Period and semimajor axis of the m1 motion is given (astrometric signature of planet host star)
    '''
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
#  *       - T0_day Time of passage at periastron (julian date-2400000)
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


def astrom_signal(t_day,psi_deg,ecc,P_day,T0_day,TIC):  
    #USAGE of pseudo eccentricity
    # a = [pecc,P_day,T0_day,A,B,F,G]
    # input: xp = structure containing dates and baseline orientations of measurements
    #         a = structure containing aric orbit parameters
    # output: phi = displacment angle in mas
    # pecc = a(1)    #ecc = abs(double(atan(pecc)*2/pi))    # ecc = retrEcc( pecc )

    # psi_rad = psi_deg *2*np.pi/360
    psi_rad = np.deg2rad(psi_deg)
  
      
    # compute eccentric anomaly
    E_rad = eccentric_anomaly(ecc,t_day,T0_day,P_day)
 
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


def astrom_signalFast(t_day,spsi,cpsi,ecc,P_day,T0_day,TIC):  
        
    # compute eccentric anomaly
    E_rad = eccentric_anomaly(ecc,t_day,T0_day,P_day)
 
    # compute orbit projected on the sky
    if np.all(ecc == 0):
        X = np.cos(E_rad)
        Y = np.sin(E_rad)
    else:
        X = np.cos(E_rad)-ecc
        Y = np.sqrt(1.-ecc**2)*np.sin(E_rad)
        
    # pdb.set_trace()
    phi = (TIC[0]*spsi+TIC[1]*cpsi)*X + (TIC[2]*spsi + TIC[3]*cpsi)*Y
    # phi = np.multiply((np.multiply(TIC[0],spsi)+np.multiply(TIC[1],cpsi)),X) + np.multiply((np.multiply(TIC[2],spsi) + np.multiply(TIC[3],cpsi)),Y)
    # return np.array(phi)
    return phi


def readEphemeris(horizons_file_seed):
    
    ephDir = '/Users/jsahlmann/astro/palta/processing/data/earthEphemeris/'
    
    fitsFile = ephDir + horizons_file_seed + '_XYZ.fits'
    if not os.path.isfile(fitsFile):
        ephFile = ephDir + horizons_file_seed + '.txt'
        f_rd = open(ephFile, 'r')
        file_lines = f_rd.readlines()[0].split('\r')
        f_rd.close()
        index_start = [i for i in range(len(file_lines)) if "$$SOE" in file_lines[i]][0]
        index_end   = [i for i in range(len(file_lines)) if "$$EOE" in file_lines[i]][0]
        nBlankLines = len([i for i in range(index_start) if (file_lines[i] == '' or file_lines[i] == ' ')])
        data_start = index_start - nBlankLines + 1
        data_end = data_start + index_end - index_start - 1
        xyzdata = Table.read(ephFile, format='ascii.no_header',delimiter=',',data_start = data_start,data_end=data_end,names=('JD','ISO','X','Y','Z','tmp'), guess=False,comment='mycomment99')
        xyzdata['JD','X','Y','Z'].write(fitsFile, format = 'fits')
 #        xyzdata = Table.read(ephFile, format='ascii',delimiter='\t',data_start = 0,names=('JD','X','Y','Z'))
#         xyzdata.write(fitsFile, format = 'fits')
    else:
        xyzdata = Table.read(fitsFile, format = 'fits')

    return xyzdata


def getParallaxFactors(RA_deg,DE_deg,t_JD, horizons_file_seed=None, verbose = 0, instrument=None):
    
    ephFactor = -1
    RA_rad = np.deg2rad(RA_deg)
    DE_rad = np.deg2rad(DE_deg)

    if horizons_file_seed is None:
        horizons_file_seed = 'horizons_XYZ_2009-2019_EQUATORIAL_Paranal_1h'
    
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
            xyzdata = readEphemeris( ephDict[ins] )
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
    
    else:
    
        if verbose:
            print('Getting Parallax factors using Seed: \t%s' % horizons_file_seed)

        xyzdata = readEphemeris( horizons_file_seed )

        Xip = interp1d(xyzdata['JD'],xyzdata['X'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
        Yip = interp1d(xyzdata['JD'],xyzdata['Y'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
        Zip = interp1d(xyzdata['JD'],xyzdata['Z'], kind='linear', copy=True, bounds_error=True,fill_value=np.nan)
    

        try:    
            parfRA  = ephFactor* ( Xip(t_JD)*np.sin(RA_rad) - Yip(t_JD)*np.cos(RA_rad) )
            parfDE =  ephFactor*(( Xip(t_JD)*np.cos(RA_rad) + Yip(t_JD)*np.sin(RA_rad) )*np.sin(DE_rad) - Zip(t_JD)*np.cos(DE_rad))
        except ValueError:
            print('Error in time interpolation for parallax factors: range %3.1f--%3.2f (%s--%s)' % (np.min(t_JD),np.max(t_JD), Time(np.min(t_JD),format='jd',scale='utc').iso, Time(np.max(t_JD),format='jd',scale='utc').iso ) )          
            1/0

        
        
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
   
    def set_five_parameter_coefficients(self, earth_ephemeris_file_seed=None, verbose=False, reference_epoch_MJD=None):
    
        observing_times_2D_JD = Time(self.observing_times_2D_MJD, format='mjd').jd
        
        # compute parallax factors, this is a 2xN_obs array
        observing_parallax_factors = getParallaxFactors(self.RA_deg, self.Dec_deg, observing_times_2D_JD, horizons_file_seed=earth_ephemeris_file_seed,verbose=verbose)
          
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
        self.observing_relative_time_1D_year = observing_relative_time_1D_year
 

    def set_linear_parameter_coefficients(self, earth_ephemeris_file_seed=None, verbose=False, reference_epoch_MJD=None):

        if not hasattr(self, 'five_parameter_coefficients'):
            self.set_five_parameter_coefficients(earth_ephemeris_file_seed=earth_ephemeris_file_seed, verbose=verbose, reference_epoch_MJD=reference_epoch_MJD)
            
            
        if ('fx[1]' in self.data_2D.colnames) & ('fx[2]' in self.data_2D.colnames):    
            # the VLT/FORS2 case with a DCR corrector    
            tmp_2D = self.data_2D[self.time_column_name,'fx[1]','fy[1]','fx[2]','fy[2]'] #,'RA*_mas','DE_mas','sRA*_mas','sDE_mas','OB','frame']            
        elif ('fx[1]' in self.data_2D.colnames) & ('fx[2]' not in self.data_2D.colnames):    
            # for GTC/OSIRIS, Gemini/GMOS-N/GMOS-S, VLT/HAWK-I
            tmp_2D = self.data_2D[self.time_column_name,'fx[1]','fy[1]']
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
    
    best_genome = Table.read(best_genome_file,format='ascii.basic', data_start=2, delimiter=',', guess=False)

    # if len(best_genome) != 1:
    #     1/0

    if instrument != 'FORS2':
        best_genome.remove_column('d_mas')


    if verbose:
        for i in range(len(best_genome)):
            for c in best_genome.colnames:
                print('Planet %d: %s \t %3.3f' % (i+1, c, best_genome[c][i]))
    # thiele_innes_constants = np.array([best_genome[c][0] for c in ['A','B','F','G']])
    thiele_innes_constants = np.array([best_genome[c] for c in ['A','B','F','G']])
    a_mas, omega_deg, OMEGA_deg, i_deg = get_geomElem(thiele_innes_constants)
    d_pc  = 1./ (best_genome['plx_mas'].data.data /1000.)
    P_day = best_genome['P_day'].data.data
    a_m = a_mas / 1.e3 * d_pc * AU_m
    m1_kg = m1_MS * MS_kg
    m2_kg = pjGet_m2( m1_kg, a_m, P_day )
    m2_MS = m2_kg / MS_kg
    m2_MJ = m2_kg / MJ_kg
    TRef_MJD = reference_time_MJD

    # MIKS-GA computes T0 relative to the average time
    if verbose:
        for i in range(len(best_genome)):
            print('Planet %d: Phi0 = %f' % (i,best_genome['T0_day'][i]))
            print('Planet %d: m2_MJ = %f' % (i,m2_MJ[i]))

    best_genome['T0_day'] += TRef_MJD
    
    best_genome['omega_deg'] = omega_deg
    best_genome['i_deg'] = i_deg
    best_genome['OMEGA_deg'] = OMEGA_deg
    best_genome['m1_MS'] = m1_MS
    best_genome['m2_MS'] = m2_MS

    col_list = theta_names #np.array(['P_day','ecc','m1_MS','m2_MS','omega_deg','T0_day','dRA0_mas','dDE0_mas','plx_mas','muRA_mas','muDE_mas','rho_mas','d_mas','OMEGA_deg','i_deg'])

    for i in range(len(best_genome)):
        if i == 0:
            theta_best_genome = np.array([best_genome[c][i] for c in col_list])
        else:
            theta_best_genome = np.vstack((theta_best_genome, np.array([best_genome[c][i] for c in col_list])))

    if verbose:
        for i in range(len(best_genome)):
            for c in col_list:
                print('Planet %d: Adopted: %s \t %3.3f' % (i, c, best_genome[c][i]))
 
    return theta_best_genome
