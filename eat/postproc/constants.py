import numpy as np

STATION_FROT = {'PV':(1,-1,0),'MG':(1,1,0),'SW':(1,-1,np.pi/4.),'LM': (1,-1,0),
                'AA':(1,0,0),'SZ':(1,0,0),'AX':(1,1,0),'MM':(1,0,0),'GL': (1,0,0),
                'NN':(1,0,0),'KT':(1,0,0),'SR':(1,-1,np.pi/4.),
                'GB':(1,0,0),'FD':(1,0,0),'PT':(1,0,0),'LA':(1,0,0),'KP':(1,0,0),
                'MK':(1,0,0),'BR':(1,0,0),'NL':(1,0,0),'OV':(1,0,0),'YS':(1,0,0),'EB':(1,0,0),
                'AP':(1,1,0),'AZ':(1,1,0),'JC':(1,0,0),'SM':(1,-1,np.pi/4.),'SP':(1,0,0),'SR':(1,-1,np.pi/4.)
                }

BLTYPE = [('time','f8'),('t1','a32'),('t2','a32')]
DTCAL = [('time','f8'), ('rscale','c16'), ('lscale','c16')]

MJD_0 = 2400000.5
SECONDS_IN_DAY = 86400.0  # Number of seconds in a day

# Comment out the following legacy constants not being used anywhere currently
# ----------------------------------------------------------------------------
# DTARR = [('site', 'a32'), ('x','f8'), ('y','f8'), ('z','f8')]
# DTPOL = [('time','f8'),('freq','f8'),('tint','f8'),
#             ('t1','a32'),('t2','a32'),
#             ('u','f8'),('v','f8'),
#             ('rr','c16'),('ll','c16'),('rl','c16'),('lr','c16'),
#             ('rrweight','f8'),('llweight','f8'),('rlweight','f8'),('lrweight','f8')]
# HOUR = 15.0*(np.pi/180.0)  # 15 degrees in radians
# EP = 1.e-5
# CORRCOEFF = 10000.0
# C = 299792458.0
# MHZ2HZ = 1e6
# RADPERARCSEC = (np.pi / 180.) / 3600.
# the following copied from hops2uvfits.py
# RDATE = '2017-04-04' #reference date
# rdate_tt = Time(RDATE, format='isot', scale='utc')
# RDATE_JD = rdate_tt.jd
# RDATE_GSTIA0 = rdate_tt.sidereal_time('apparent','greenwich').degree
# RDATE_OFFSET = rdate_tt.ut1.datetime.second - rdate_tt.utc.datetime.second
# RDATE_OFFSET += 1.e-6*(rdate_tt.ut1.datetime.microsecond - rdate_tt.utc.datetime.microsecond)