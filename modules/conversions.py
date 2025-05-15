## This file was never used in the final product, however it took a lot of work so I would like 
## to include it alongside my submission!
import math
import numpy as np
#https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
# https://ssd.jpl.nasa.gov/astro_par.html
class AstroConstants:
    R_EARTH_KM = 6378 # radius of the Earth
    R_SUN_KM = 696340 # radius of the Sun
    RATIO_EARTH_SUN = R_EARTH_KM / R_SUN_KM # ratio of the earth to the sun
    SIGMA = 5.67*1e-8
    '''The Stefan-Boltzman constant (W / m[2]•K[4]).'''
    G = 6.67430e-11
    '''The universal gravitational constant, G.'''
    AU = 149597870.7
    '''1 astronomical unit: the distance between the Earth and the Sun in kilometres.'''

def approx_orbital_distance(pl_orbper, st_rad, st_logg):
    '''Calculates the approximate orbital distance of a planet in astronomical units, 
    using Kepler's third law of planetary motion. Assumes orbital period in days, star radius in R[sun] and star surface
    gravity in cgs units, log g. Roughly 7% off the true value '''

    # The formula for calculating orbital distance given these three parameters in the specified units has been
    # significantly reworked to avoid numerical overflow.
    # This calculates the orbital distance fr
    ## T^2 = (4π^2 / GM) * r^3 --> r = (G*M*T^2 / 4π^2)^2/3
    k = (6.96340 * 8.640) / (2 * math.pi) # converts st_rad to meters, pl_orbper to seconds 
    # 1 day = 8.640 * 10^4 seconds -> T^2 [s] = (8.64 * 10^4)^2 * T^2
    # R[sun] = 696340 km = 6.96340 * 10^8 m -> R^2 [m] = (6.96340 * 10^8)^2 * R^2
    # g is in log(g) cgs units (cm^2 / s) -> g [m/s^2] = 10^(g - 2) (exponent laws)
    x = 5 + (st_logg - 2)/3 ## (10^log(g) / 10^2 [cm / m])^1/3 -> 8 comes from (10^(24/3))
    r = np.float_power(k, 2/3) * np.float_power(pl_orbper, 2/3) * np.float_power(st_rad, 2/3) * np.float_power(10, x)
    
    return r / AstroConstants.AU

def stellar_mass(st_rad, st_logg):
    g = math.pow(10, st_logg) / 100 # convert cm / s^2 to m / s^2
    m = g * math.pow(st_rad, 2) / AstroConstants.G
    return m

def planet_star_ratio_squared(t_depth_ppm):
    return t_depth_ppm * 1e-6 

def planet_star_ratio(t_depth_ppm):
    return math.sqrt(t_depth_ppm) * 1e-3 # 1 / sqrt(10e6) = 10e3

def inverse_planet_star_ratio(ratio):
    return math.pow(ratio, 2)*1e6