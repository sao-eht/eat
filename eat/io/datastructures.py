class Caltable(object):
    """
    Represents a polarimetric VLBI observation of visibility amplitudes and phases.
    A minimal version of the Caltable class in eht-imaging.

    Parameters
    ----------
    ra : float
        Right ascension of the source (degrees).
    dec : float
        Declination of the source (degrees).
    rf : float
        Reference frequency (GHz).
    bw : float
        Bandwidth (GHz).
    datatables : list
        List of data tables containing visibility amplitudes and phases.
    tarr : list
        List of dictionaries with site information.
    source : str, optional
        Name of the source. Default is 'NONE'.
    mjd : int, optional
        Modified Julian Date of the observation. Default is 0.
    timetype : {'GMST', 'UTC'}, optional
        Time type for the observation. Default is 'UTC'.

    Attributes
    ----------
    source : str
        Name of the source.
    ra : float
        Right ascension of the source.
    dec : float
        Declination of the source.
    rf : float
        Reference frequency.
    bw : float
        Bandwidth.
    mjd : int
        Modified Julian Date.
    timetype : str
        Time type for the observation.
    tarr : list
        List of site information.
    tkey : dict
        Dictionary mapping site names to array indices.
    data : list
        List of data tables containing visibility amplitudes and phases.

    Methods
    -------
    copy()
        Returns a copy of the Caltable object.
    """

    def __init__(self, ra, dec, rf, bw, datatables, tarr, source='NONE', mjd=0, timetype='UTC'):
        """
        A polarimetric VLBI observation of visibility amplitudes and phases (in Jy).

        Args:
            ra (float): Right ascension of the source (degrees).
            dec (float): Declination of the source (degrees).
            rf (float): Reference frequency (GHz).
            bw (float): Bandwidth (GHz).
            datatables (list): List of data tables containing visibility amplitudes and phases.
            tarr (list): List of dictionaries with site information.
            source (str, optional): Name of the source. Default is 'NONE'.
            mjd (int, optional): Modified Julian Date of the observation. Default is 0.
            timetype (str, optional): Time type for the observation. Default is 'UTC'.

           Returns:
               caltable (Caltable): an Caltable object
        """

        if len(datatables) == 0:
            raise Exception("No data in input table!")

        # Set the various parameters
        self.source = str(source)
        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.bw = float(bw)
        self.mjd = int(mjd)

        if timetype not in ['GMST', 'UTC']:
            raise Exception("timetype must by 'GMST' or 'UTC'")
        self.timetype = timetype
        self.tarr = tarr

        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

        # Save the data
        self.data = datatables

    def copy(self):
        """
        Create a copy of the Caltable object.

        Returns
        -------
        Caltable
            A copy of the current Caltable object.
        """
        new_caltable = Caltable(self.ra, self.dec, self.rf, self.bw, self.data, self.tarr, source=self.source, mjd=self.mjd, timetype=self.timetype)
        return new_caltable

class Uvfits_data(object):
    """
    Class to store data table and random group parameter arrays for saving to UVFITS format.

    Parameters
    ----------
    u : array-like
        Array of u coordinates (spatial frequency component).
    v : array-like
        Array of v coordinates (spatial frequency component).
    bls : array-like
        Array of baseline identifiers.
    jds : array-like
        Array of Julian dates corresponding to the observations.
    tints : array-like
        Array of integration times for each observation.
    datatable : array-like
        Data table containing visibility and related information.

    Attributes
    ----------
    u : array-like
        Stored u coordinates.
    v : array-like
        Stored v coordinates.
    bls : array-like
        Stored baseline identifiers.
    jds : array-like
        Stored Julian dates.
    tints : array-like
        Stored integration times.
    datatable : array-like
        Stored data table.
    """
    def __init__(self, u, v, bls, jds, tints, datatable):
        self.u = u
        self.v = v
        self.bls = bls
        self.jds = jds
        self.tints = tints
        self.datatable = datatable

class Obs_info(object):
    """
    Obs_info stores observational metadata for a given source.
    Parameters
    ----------
    src : str
        Name of the observed source.
    ra : float
        Right ascension of the source (in degrees or appropriate units).
    dec : float
        Declination of the source (in degrees or appropriate units).
    ref_freq : float
        Reference frequency for the observation (in Hz).
    ch_bw : float
        Channel bandwidth (in Hz).
    ch_spacing : float
        Spacing between channels (in Hz).
    ch_1 : float
        Frequency of the first channel (in Hz).
    nchan : int
        Number of frequency channels.
    scan_array : array-like
        Array containing scan information for the observation.
    Attributes
    ----------
    src : str
        Name of the observed source.
    ra : float
        Right ascension of the source.
    dec : float
        Declination of the source.
    ref_freq : float
        Reference frequency for the observation.
    ch_bw : float
        Channel bandwidth.
    ch_spacing : float
        Channel spacing.
    ch_1 : float
        Frequency of the first channel.
    nchan : int
        Number of channels.
    scans : array-like
        Scan information array.
    """
    def __init__(self, src, ra, dec, ref_freq, ch_bw, ch_spacing, ch_1, nchan, scan_array):
        self.src = src
        self.ra = ra
        self.dec = dec
        self.ref_freq = ref_freq
        self.ch_bw = ch_bw
        self.ch_spacing = ch_spacing
        self.ch_1 = ch_1
        self.nchan = nchan
        self.scans = scan_array

class Antenna_info(object):
    """
    Antenna metadata class to store information about antennas.
    Parameters
    ----------
    antnames : list of str
        List of antenna names.
    antnums : list of int
        List of antenna numbers or identifiers.
    xyz : array-like, shape (N, 3)
        Array containing the (x, y, z) coordinates for each antenna.
    Attributes
    ----------
    antnames : list of str
        Names of the antennas.
    antnums : list of int
        Numbers or identifiers of the antennas.
    xyz : array-like
        (x, y, z) coordinates of the antennas.
    """
    def __init__(self, antnames, antnums, xyz):
        self.antnames =  antnames
        self.antnums = antnums
        self.xyz = xyz

class Datastruct(object):
    """
    Data and metadata container for saving to UVFITS format.

    Parameters
    ----------
    obs_info : object
        Observation information.
    antenna_info : object or table
        Antenna information. In UVFITS format, this is an Antenna_info object.
        In ehtim format, this is a table.
    data : object or table
        Data table. In UVFITS format, this is a Uvfits_data object.
        In ehtim format, this is a table.
    dtype : str, optional
        Format type of the data table ('UVFITS' or 'ehtim'). Default is 'UVFITS'.

    Notes
    -----
    - In ehtim format, `antenna_info` and `data` are tables.
    - In UVFITS format, `antenna_info` and `data` are Antenna_info and Uvfits_data objects, respectively.
    """

    def __init__(self, obs_info, antenna_info, data, dtype='UVFITS'):
        self.dtype = dtype
        self.obs_info = obs_info
        self.antenna_info = antenna_info
        self.data = data