from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.coordinates import Angle

class Site:
    """
    A class describing telescope site information
    """
    def __init__(self, d):
        self.site  = d['site']
        self.coord = EarthLocation(d['x_m'], d['y_m'], d['z_m'], u.m)
        self.cnst  = Angle(d['fra_offset'], u.degree)
        self.sign  = d['fra_elev_sign']
