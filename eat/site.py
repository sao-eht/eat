import numpy as np

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.coordinates import Angle
from astropy.coordinates import AltAz

def par_angle(hra, dec, lat):
    y = np.sin(hra) * np.cos(lat)
    x = np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(hra)
    return np.arctan2(y, x)

class Site:
    """
    A class describing telescope site information
    """
    def __init__(self, d):
        self.site  = d['site']
        self.coord = EarthLocation(d['x_m'], d['y_m'], d['z_m'], u.m)
        self.cnst  = Angle(d['fra_offset'], u.degree)
        self.sign  = d['fra_elev_sign']

    def fra(self, time, src_coord):
        aa = src_coord.transform_to(AltAz(obstime=time, location=self.coord))
        pa = par_angle(aa.az, src_coord.dec, self.coord.lat)
        return pa + self.cnst + self.sign * aa.alt
