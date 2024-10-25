"""provides python access to mk4 data-types via c-library"""

#core imports
from __future__ import print_function
from builtins import range
import ctypes
import os

################################################################################
# general records and utils
################################################################################

def mk4redefine_array_length(array, new_size):
    """ This function is nasty hack needed for redefining the size of a variable
    length array this mimics the c-structs ability to ignore bounds checking.
    Otherwise python/ctypes complains that we are accessing data at an invalid
    index and wont let us get at the data.
    For example, this can be used for accessing data beyond the first
    element in the 'data' field of type_212, by doing something like: (x is a type_212 ptr)
    accessible_data = mk4redefine_array_length(x.contents.data, x.contents.nap ) """
    return (array._type_*new_size).from_address(ctypes.addressof(array))

def mk4redefine_char_array_length(array, new_size):
    """same as mk4redefine_array_length, but specific to char arrays (strings)"""
    return (ctypes.c_char*new_size).from_address(ctypes.addressof(array))


def mk4fp_approximately_equal(a, b, abs_tol=1e-14, rel_tol=1e-6):
    """simple floating point comparison"""
    return abs(a-b) <= max( abs( rel_tol*max( abs(a), abs(b) ) ), abs(abs_tol) )


class Mk4StructureBase(ctypes.Structure):
    """mk4 base class which implements comparison operations eq and ne and a print-summary"""

    def __eq__(self, other):
        for field in self._fields_:
            a, b = getattr(self, field[0]), getattr(other, field[0])
            if isinstance(a, ctypes.Array):
                if a[:] != b[:]:
                    return False
            else:
                if a != b:
                    return False
        return True

    def __ne__(self, other):
        for field in self._fields_:
            a, b = getattr(self, field[0]), getattr(other, field[0])
            if isinstance(a, ctypes.Array):
                if a[:] != b[:]:
                    return True
            else:
                if a != b:
                    return True
        return False

    def printsummary(self):
        """dump data summary"""
        print(self.__class__.__name__, ":")
        for field in self._fields_:
            a = getattr(self, field[0])
            if isinstance(a, ctypes.Array):
                print(field[0], ":", "array of length: ", len(a), ":")
                for x in a:
                    if isinstance(x, Mk4StructureBase):
                        x.printsummary()
                    else:
                        print(x)
            elif isinstance(a, Mk4StructureBase):
                print(field[0], ":")
                a.printsummary()
            else:
                print(field[0], ":" , a)

    #probably superfluous, most types seem to contain floats
    def contains_floats(self):
        for field in self._fields_:
            a = getattr(self, field[0])
            if isinstance(a, float):
                return True
            elif isinstance(a, Mk4StructureBase):
                if a.contains_floats():
                    return True
            elif isinstance(a, ctypes.Array) and len(a) > 0:
                if isinstance(a[0], Mk4StructureBase):
                    if a[0].contains_floats():
                        return True
                elif isinstance(a[0], float):
                    return True
        return False

    def approximately_equal(self, other, ignore_dates=True, verbose=True, abs_tol=1e-14, rel_tol=1e-6):
        """ compares two object, wary about floating point types """
        if ignore_dates and isinstance(self,date):
            return True
        for field in self._fields_:
            a, b = getattr(self, field[0]), getattr(other, field[0])
            #ignore various chunks of unused data which may be uninitialized
            #ignore 'sample_rate'in the type_203 records also, since it is populated with a junk value
            if not (field[0] == 'unused' or field[0] == 'unused1' or field[0] == 'unused2' or field[0] == 'unused3' or field[0] == 'sample_rate'):
                if isinstance(a, ctypes.Array):
                    if isinstance(a[0], Mk4StructureBase):
                        for i in list(range(len(a))):
                            if not a[i].approximately_equal(b[i], ignore_dates, abs_tol, rel_tol):
                                if verbose:
                                    print("within type: ", self.__class__.__name__, "at index: ", i)
                                return False
                    elif isinstance(a[0], float):
                        for i in list(range(len(a))):
                            if not mk4fp_approximately_equal(a[i], b[i], abs_tol, rel_tol):
                                if verbose:
                                    print("field:", field[0], "at index: ", i, "is not equal in: ", self.__class__.__name__ , "(", a, " != ", b, ")")
                                return False
                elif isinstance(a, Mk4StructureBase):
                    if not a.approximately_equal(b, ignore_dates, abs_tol, rel_tol):
                        if verbose:
                            print("within type: ", self.__class__.__name__)
                        return False
                elif isinstance(a, float):
                    if not mk4fp_approximately_equal(a, b, abs_tol, rel_tol):
                        if verbose:
                            print("field:", field[0], "is not equal in: ", self.__class__.__name__ , "(", a, " != ", b, ")")
                        return False
                else:
                    if a != b:
                        if verbose:
                            print("field:", field[0], "is not equal in: ", self.__class__.__name__ , "(", a, " != ", b, ")")
                        return False
        return True


class type_000(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('date', ctypes.c_char * 16),
        ('name', ctypes.c_char * 40),
    ]

class date(Mk4StructureBase):
    _fields_ = [
        ('year', ctypes.c_short),
        ('day', ctypes.c_short),
        ('hour', ctypes.c_short),
        ('minute', ctypes.c_short),
        ('second', ctypes.c_float),
    ]

################################################################################
# type_1XX records (corel files)
################################################################################

class type_100(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('procdate', date),
        ('baseline', ctypes.c_char * 2),
        ('rootname', ctypes.c_char * 34),
        ('qcode', ctypes.c_char * 2),
        ('unused2', ctypes.c_char * 6),
        ('pct_done', ctypes.c_float),
        ('start', date),
        ('stop', date),
        ('ndrec', ctypes.c_int),
        ('nindex', ctypes.c_int),
        ('nlags', ctypes.c_short),
        ('nblocks', ctypes.c_short),
    ]

class type_101(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('status', ctypes.c_char),
        ('nblocks', ctypes.c_short),
        ('index', ctypes.c_short),
        ('primary', ctypes.c_short),
        ('ref_chan_id', ctypes.c_char * 8),
        ('rem_chan_id', ctypes.c_char * 8),
        ('corr_board', ctypes.c_short),
        ('corr_slot', ctypes.c_short),
        ('ref_chan', ctypes.c_short),
        ('rem_chan', ctypes.c_short),
        ('post_mortem', ctypes.c_int),
        ('blocks', ctypes.c_int * 1),
    ]


class flag_wgt(ctypes.Union):
    _fields_ = [
        ('flag', ctypes.c_int),
        ('weight', ctypes.c_float),
    ]

class counts_per_lag(Mk4StructureBase):
    _fields_ = [
        ('coscor', ctypes.c_int),
        ('cosbits', ctypes.c_int),
        ('sincor', ctypes.c_int),
        ('sinbits', ctypes.c_int),
    ]

class lag_tag(Mk4StructureBase):
    _fields_ = [
        ('coscor', ctypes.c_int),
        ('sincor', ctypes.c_int),
    ]

class counts_global(Mk4StructureBase):
    _fields_ = [
        ('cosbits', ctypes.c_int),
        ('sinbits', ctypes.c_int),
        ('lags', lag_tag * 1),
    ]

class auto_per_lag(Mk4StructureBase):
    _fields_ = [
        ('coscor', ctypes.c_int),
        ('cosbits', ctypes.c_int),
    ]

class auto_global(Mk4StructureBase):
    _fields_ = [
        ('cosbits', ctypes.c_int),
        ('unused', ctypes.c_char * 4),
        ('coscor', ctypes.c_int * 1),
    ]

class spectral(Mk4StructureBase):
    _fields_ = [
        ('re', ctypes.c_float),
        ('im', ctypes.c_float),
    ]

class lag_data(ctypes.Union):
    _fields_ = [
        ('cpl', counts_per_lag * 1),
        ('cg', counts_global),
        ('apl', auto_per_lag * 1),
        ('ag', auto_global),
        ('spec', spectral * 1),
    ]

class type_120(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('type', ctypes.c_char),
        ('nlags', ctypes.c_short),
        ('baseline', ctypes.c_char * 2),
        ('rootcode', ctypes.c_char * 6),
        ('index', ctypes.c_int),
        ('ap', ctypes.c_int),
        ('fw', flag_wgt),
        ('status', ctypes.c_int),
        ('fr_delay', ctypes.c_int),
        ('delay_rate', ctypes.c_int),
        ('ld', lag_data),
    ]

    def __getattribute__(self,name):
        """ another nasty kludge to get around the apparent (but not really) fixed size of the 'lag_data' array
            also handles some of the wierd behavior in the way the length of the array is treated
            depending on the union type, and returns an object array cast to the union's actual underlying type """
        if name == 'ld' and object.__getattribute__(self, 'type') == '\x01': #COUNTS_PER_LAG format
            return mk4redefine_array_length(object.__getattribute__(self, 'ld').cpl, object.__getattribute__(self, 'nlags'))
        if name == 'ld' and object.__getattribute__(self, 'type') == '\x02': #COUNTS_GLOBAL format
            return mk4redefine_array_length(object.__getattribute__(self, 'ld').cg, 1)
        if name == 'ld' and object.__getattribute__(self, 'type') == '\x03': #AUTO_GLOBAL format
            return mk4redefine_array_length(object.__getattribute__(self, 'ld').apl, 1)
        if name == 'ld' and object.__getattribute__(self, 'type') == '\x04': #AUTO_PER_LAG format
            return mk4redefine_array_length(object.__getattribute__(self, 'ld').ag, object.__getattribute__(self, 'nlags'))
        if name == 'ld' and object.__getattribute__(self, 'type') == '\x05': #SPECTRAL format
            return mk4redefine_array_length(object.__getattribute__(self, 'ld').spec, object.__getattribute__(self, 'nlags'))
        if name == 'fw' and object.__getattribute__(self, 'type') == '\x01': #COUNTS_PER_LAG format
            return object.__getattribute__(self, 'fw').flag
        if name == 'fw' and object.__getattribute__(self, 'type') == '\x02': #COUNTS_GLOBAL format
            return object.__getattribute__(self, 'fw').flag
        if name == 'fw' and object.__getattribute__(self, 'type') == '\x03': #AUTO_GLOBAL format
            return object.__getattribute__(self, 'fw').flag
        if name == 'fw' and object.__getattribute__(self, 'type') == '\x04': #AUTO_PER_LAG format
            return object.__getattribute__(self, 'fw').flag
        if name == 'fw' and object.__getattribute__(self, 'type') == '\x05': #SPECTRAL format
            return object.__getattribute__(self, 'fw').weight
        else:
            return object.__getattribute__(self, name)

class index_tag(Mk4StructureBase):
    _fields_ = [
        ('t101', ctypes.POINTER(type_101)),
        ('ap_space', ctypes.c_int),
        ('t120', ctypes.POINTER(ctypes.POINTER(type_120))),
    ]

class mk4_corel(Mk4StructureBase):
    _fields_ = [
        ('allocated', ctypes.POINTER(None) * (8192 + 4)),
        ('nalloc', ctypes.c_int),
        ('file_image', ctypes.POINTER(ctypes.c_char)),
        ('id', ctypes.POINTER(type_000)),
        ('t100', ctypes.POINTER(type_100)),
        ('index_space', ctypes.c_int),
        ('index', ctypes.POINTER(index_tag)),
    ]

    def __del__(self):
        """this is needed to prevent memory leaks in the underlying c-library"""
        dfio = mk4io_load()
        dfio.clear_mk4corel(ctypes.byref(self))

################################################################################
# type_2XX records (fringe files)
################################################################################

class sky_coord(Mk4StructureBase):
    _fields_ = [
        ('ra_hrs', ctypes.c_short),
        ('ra_mins', ctypes.c_short),
        ('ra_secs', ctypes.c_float),
        ('dec_degs', ctypes.c_short),
        ('dec_mins', ctypes.c_short),
        ('dec_secs', ctypes.c_float),
    ]

class type_200(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('software_rev', ctypes.c_short * 10),
        ('expt_no', ctypes.c_int),
        ('exper_name', ctypes.c_char * 32),
        ('scan_name', ctypes.c_char * 32),
        ('correlator', ctypes.c_char * 8),
        ('scantime', date),
        ('start_offset', ctypes.c_int),
        ('stop_offset', ctypes.c_int),
        ('corr_date', date),
        ('fourfit_date', date),
        ('frt', date),
    ]

class type_201(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('source', ctypes.c_char * 32),
        ('coord', sky_coord),
        ('epoch', ctypes.c_short),
        ('unused2', ctypes.c_char * 2),
        ('coord_date', date),
        ('ra_rate', ctypes.c_double),
        ('dec_rate', ctypes.c_double),
        ('pulsar_phase', ctypes.c_double * 4),
        ('pulsar_epoch', ctypes.c_double),
        ('dispersion', ctypes.c_double),
    ]

class type_202(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('baseline', ctypes.c_char * 2),
        ('ref_intl_id', ctypes.c_char * 2),
        ('rem_intl_id', ctypes.c_char * 2),
        ('ref_name', ctypes.c_char * 8),
        ('rem_name', ctypes.c_char * 8),
        ('ref_tape', ctypes.c_char * 8),
        ('rem_tape', ctypes.c_char * 8),
        ('nlags', ctypes.c_short),
        ('ref_xpos', ctypes.c_double),
        ('rem_xpos', ctypes.c_double),
        ('ref_ypos', ctypes.c_double),
        ('rem_ypos', ctypes.c_double),
        ('ref_zpos', ctypes.c_double),
        ('rem_zpos', ctypes.c_double),
        ('u', ctypes.c_double),
        ('v', ctypes.c_double),
        ('uf', ctypes.c_double),
        ('vf', ctypes.c_double),
        ('ref_clock', ctypes.c_float),
        ('rem_clock', ctypes.c_float),
        ('ref_clockrate', ctypes.c_float),
        ('rem_clockrate', ctypes.c_float),
        ('ref_idelay', ctypes.c_float),
        ('rem_idelay', ctypes.c_float),
        ('ref_zdelay', ctypes.c_float),
        ('rem_zdelay', ctypes.c_float),
        ('ref_elev', ctypes.c_float),
        ('rem_elev', ctypes.c_float),
        ('ref_az', ctypes.c_float),
        ('rem_az', ctypes.c_float),
    ]

class ch_struct(Mk4StructureBase):
    _fields_ = [
        ('index', ctypes.c_short),
        ('sample_rate', ctypes.c_ushort),
        ('refsb', ctypes.c_char),
        ('remsb', ctypes.c_char),
        ('refpol', ctypes.c_char),
        ('rempol', ctypes.c_char),
        ('ref_freq', ctypes.c_double),
        ('rem_freq', ctypes.c_double),
        ('ref_chan_id', ctypes.c_char * 8),
        ('rem_chan_id', ctypes.c_char * 8),
    ]

class type_203(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('channels', ch_struct * (8 * 64)),
    ]

class type_204(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('ff_version', ctypes.c_short * 2),
        ('platform', ctypes.c_char * 8),
        ('control_file', ctypes.c_char * 96),
        ('ffcf_date', date),
        ('override', ctypes.c_char * 128),
    ]

class ffit_chan_struct(Mk4StructureBase):
    _fields_ = [
        ('ffit_chan_id', ctypes.c_char),
        ('unused', ctypes.c_char),
        ('channels', ctypes.c_short * 4),
    ]

class type_205(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('utc_central', date),
        ('offset', ctypes.c_float),
        ('ffmode', ctypes.c_char * 8),
        ('search', ctypes.c_float * 6),
        ('filter', ctypes.c_float * 8),
        ('start', date),
        ('stop', date),
        ('ref_freq', ctypes.c_double),
        ('ffit_chan', ffit_chan_struct * 64),
    ]

class sidebands(Mk4StructureBase):
    _fields_ = [
        ('lsb', ctypes.c_short),
        ('usb', ctypes.c_short),
    ]

class sbweights(Mk4StructureBase):
    _fields_ = [
        ('lsb', ctypes.c_double),
        ('usb', ctypes.c_double),
    ]

class type_206(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('start', date),
        ('first_ap', ctypes.c_short),
        ('last_ap', ctypes.c_short),
        ('accepted', sidebands * 64),
        ('weights', sbweights * 64),
        ('intg_time', ctypes.c_float),
        ('accept_ratio', ctypes.c_float),
        ('discard', ctypes.c_float),
        ('reason1', sidebands * 64),
        ('reason2', sidebands * 64),
        ('reason3', sidebands * 64),
        ('reason4', sidebands * 64),
        ('reason5', sidebands * 64),
        ('reason6', sidebands * 64),
        ('reason7', sidebands * 64),
        ('reason8', sidebands * 64),
        ('ratesize', ctypes.c_short),
        ('mbdsize', ctypes.c_short),
        ('sbdsize', ctypes.c_short),
        ('unused2', ctypes.c_char * 6),
    ]

class sbandf(Mk4StructureBase):
    _fields_ = [
        ('lsb', ctypes.c_float),
        ('usb', ctypes.c_float),
    ]

class type_207(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('pcal_mode', ctypes.c_int),
        ('unused2', ctypes.c_int),
        ('ref_pcamp', sbandf * 64),
        ('rem_pcamp', sbandf * 64),
        ('ref_pcphase', sbandf * 64),
        ('rem_pcphase', sbandf * 64),
        ('ref_pcoffset', sbandf * 64),
        ('rem_pcoffset', sbandf * 64),
        ('ref_pcfreq', sbandf * 64),
        ('rem_pcfreq', sbandf * 64),
        ('ref_pcrate', ctypes.c_float),
        ('rem_pcrate', ctypes.c_float),
        ('ref_errate', ctypes.c_float * 64),
        ('rem_errate', ctypes.c_float * 64),
    ]

class type_208(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('quality', ctypes.c_char),
        ('errcode', ctypes.c_char),
        ('tape_qcode', ctypes.c_char * 6),
        ('adelay', ctypes.c_double),
        ('arate', ctypes.c_double),
        ('aaccel', ctypes.c_double),
        ('tot_mbd', ctypes.c_double),
        ('tot_sbd', ctypes.c_double),
        ('tot_rate', ctypes.c_double),
        ('tot_mbd_ref', ctypes.c_double),
        ('tot_sbd_ref', ctypes.c_double),
        ('tot_rate_ref', ctypes.c_double),
        ('resid_mbd', ctypes.c_float),
        ('resid_sbd', ctypes.c_float),
        ('resid_rate', ctypes.c_float),
        ('mbd_error', ctypes.c_float),
        ('sbd_error', ctypes.c_float),
        ('rate_error', ctypes.c_float),
        ('ambiguity', ctypes.c_float),
        ('amplitude', ctypes.c_float),
        ('inc_seg_ampl', ctypes.c_float),
        ('inc_chan_ampl', ctypes.c_float),
        ('snr', ctypes.c_float),
        ('prob_false', ctypes.c_float),
        ('totphase', ctypes.c_float),
        ('totphase_ref', ctypes.c_float),
        ('resphase', ctypes.c_float),
        ('tec_error', ctypes.c_float),
    ]

class polars(Mk4StructureBase):
    _fields_ = [
        ('ampl', ctypes.c_float),
        ('phase', ctypes.c_float),
    ]

class type_210(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('amp_phas', polars * 64),
    ]

class phasor(Mk4StructureBase):
    _fields_ = [
        ('amp', ctypes.c_float),
        ('phase', ctypes.c_float),
    ]

class newphasor(Mk4StructureBase):
    _fields_ = [
        ('amp', ctypes.c_float),
        ('phase', ctypes.c_float),
        ('weight', ctypes.c_float),
    ]

class type_212(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused', ctypes.c_char),
        ('nap', ctypes.c_short),
        ('first_ap', ctypes.c_short),
        ('channel', ctypes.c_short),
        ('sbd_chan', ctypes.c_short),
        ('unused2', ctypes.c_char * 2),
        ('data', newphasor * 1),
    ]

    def __getattribute__(self,name):
        """nasty kludge to get around the apparent fixed size of the 'data' array"""
        if name == 'data':
            return mk4redefine_array_length(object.__getattribute__(self, 'data'), object.__getattribute__(self, 'nap'))
        else:
            return object.__getattribute__(self, name)

class type_220(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('width', ctypes.c_short),
        ('height', ctypes.c_short),
        ('fplot', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
    ]

class mutable_length_char_array(ctypes.Structure):
    """ this dummy class is used to get access to the char arrays in type_221 and type_222
    this is just straight up abuse of ctypes and should be frowned upon """
    _fields_ = []

class type_221(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char),
        ('padded', ctypes.c_short),
        ('ps_length', ctypes.c_int),
        ('pplot', mutable_length_char_array),
    ]

    def __getattribute__(self,name):
        """nasty kludge to get access to 'pplot' char array"""
        if name == 'pplot':
            return ctypes.string_at( ctypes.byref(object.__getattribute__(self, 'pplot') ), object.__getattribute__(self, 'ps_length') )
        else:
            return object.__getattribute__(self, name)

class type_222(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char),
        ('padded', ctypes.c_short),
        ('setstring_hash', ctypes.c_uint),
        ('control_hash', ctypes.c_uint),
        ('setstring_length', ctypes.c_int),
        ('cf_length', ctypes.c_int),
        ('control_contents', mutable_length_char_array )
    ]

    def get_setstring_padded_length(self):
        """ get the 8 byte padded length of the set-command string """
        ss_len = object.__getattribute__(self, 'setstring_length')
        return ( (ss_len + 7 ) & ~7 ) + 8

    def get_control_file_padded_length(self):
        """ get the 8 byte padded length of the control file """
        cf_len = object.__getattribute__(self, 'cf_length')
        return ( (cf_len + 7 ) & ~7 ) + 8

    def get_setstring(self):
        """ get the string associated with the fourfit command line 'set' option """
        return ctypes.string_at( ctypes.byref(object.__getattribute__(self, 'control_contents')), self.setstring_length)

    def get_control_file_contents(self):
        """ get contents of the control file as string """
        ss_pad = self.get_setstring_padded_length()
        return ctypes.string_at( ctypes.byref(object.__getattribute__(self, 'control_contents'), ss_pad), self.cf_length)

    def __getattribute__(self,name):
        """another horrible kludge to get access to the 'control_contents' array"""
        if name == 'control_contents':
            #calculate the (padded) size of the array
            full_size = self.get_setstring_padded_length() + self.get_control_file_padded_length()
            #return ctypes.string_at( object.__getattribute__(self, 'control_contents'), full_size)
            return ctypes.string_at( ctypes.byref(object.__getattribute__(self, 'control_contents') ), full_size)
        else:
            return object.__getattribute__(self, name)

# NOTE: check type of xpower, guessing for complex_struct
class complex_struct(Mk4StructureBase):
    _fields_ = [
        ('re', ctypes.c_double),
        ('im', ctypes.c_double),
    ]

# NOTE: check type of xpower, guessing for complex_struct
class type_230(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char),
        ('nspec_pts', ctypes.c_short),
        ('frq', ctypes.c_int),
        ('ap', ctypes.c_int),
        ('lsbweight', ctypes.c_float),
        ('usbweight', ctypes.c_float),
        ('xpower', complex_struct * 1),
    ]

    def __getattribute__(self,name):
        if name == 'xpower':
            return mk4redefine_array_length(object.__getattribute__(self, 'xpower'), object.__getattribute__(self, 'nspec_pts'))
        else:
            return object.__getattribute__(self, name)

class mk4_fringe(Mk4StructureBase):
    _fields_ = [
        ('allocated', ctypes.POINTER(None) * (64 + 16)),
        ('nalloc', ctypes.c_int),
        ('file_image', ctypes.POINTER(ctypes.c_char)),
        ('id', ctypes.POINTER(type_000)),
        ('t200', ctypes.POINTER(type_200)),
        ('t201', ctypes.POINTER(type_201)),
        ('t202', ctypes.POINTER(type_202)),
        ('t203', ctypes.POINTER(type_203)),
        ('t204', ctypes.POINTER(type_204)),
        ('t205', ctypes.POINTER(type_205)),
        ('t206', ctypes.POINTER(type_206)),
        ('t207', ctypes.POINTER(type_207)),
        ('t208', ctypes.POINTER(type_208)),
        ('t210', ctypes.POINTER(type_210)),
        ('n212', ctypes.c_int),
        ('t212', ctypes.POINTER(type_212) * 64),
        ('t220', ctypes.POINTER(type_220)),
        ('t221', ctypes.POINTER(type_221)),
        ('t222', ctypes.POINTER(type_222)),
        ('n230', ctypes.c_int),
        ('t230', ctypes.POINTER(type_230) * (64 * 8192)),
    ]


    def __del__(self):
        """make sure we clean up the memory allocated by the mk4 c library
        or we will have a memory leak"""
        dfio = mk4io_load()
        dfio.clear_mk4fringe(ctypes.byref(self))

################################################################################
# type_3XX records (sdata files)
################################################################################

class type_300(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 2),
        ('SU_number', ctypes.c_ubyte),
        ('id', ctypes.c_char),
        ('intl_id', ctypes.c_char * 2),
        ('name', ctypes.c_char * 32),
        ('unused2', ctypes.c_char),
        ('model_start', date),
        ('model_interval', ctypes.c_float),
        ('nsplines', ctypes.c_short)
    ]

class type_301(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('interval', ctypes.c_short),
        ('chan_id', ctypes.c_char * 32),
        ('unused2', ctypes.c_char * 6),
        ('delay_spline', ctypes.c_double * 6)
    ]

class type_302(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('interval', ctypes.c_short),
        ('chan_id', ctypes.c_char * 32),
        ('unused2', ctypes.c_char * 6),
        ('phase_spline', ctypes.c_double * 6)
    ]

class type_303(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('interval', ctypes.c_short),
        ('chan_id', ctypes.c_char * 32),
        ('unused2', ctypes.c_char * 6),
        ('azimuth', ctypes.c_double * 6),
        ('elevation', ctypes.c_double * 6),
        ('parallactic_angle', ctypes.c_double * 6),
        ('u', ctypes.c_double * 6),
        ('v', ctypes.c_double * 6),
        ('w', ctypes.c_double * 6)
    ]

class trackstat(Mk4StructureBase):
    _fields_ = [
        ('error_count', ctypes.c_int),
        ('frames', ctypes.c_int),
        ('bad_frames', ctypes.c_int),
        ('slip_sync', ctypes.c_int),
        ('missing_sync', ctypes.c_int),
        ('crc_error', ctypes.c_int)
    ]

class type_304(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('time', date),
        ('duration', ctypes.c_float),
        ('trackstats', trackstat * 64)
    ]

class type_305(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3)
    ]

class stcount_struct(Mk4StructureBase):
    _fields_ = [
        ('chan_id', ctypes.c_char * 32),
        ('bigpos', ctypes.c_int),
        ('pos', ctypes.c_int),
        ('neg', ctypes.c_int),
        ('bigneg', ctypes.c_int)
    ]

class type_306(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('time', date),
        ('duration', ctypes.c_float),
        ('stcount', stcount_struct * 16)
    ]

class ChanCount(Mk4StructureBase):
    _fields_ = [
        ('count', ctypes.c_uint * 8),
        ('val_count', ctypes.c_uint)
    ]

class type_307(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('su', ctypes.c_int),
        ('unused2', ctypes.c_char * 4),
        ('tot', ctypes.c_double),
        ('rot', ctypes.c_double),
        ('accum_period', ctypes.c_double),
        ('frame_count', ctypes.c_uint),
        ('counts', ChanCount * 16 ),
        ('unused3', ctypes.c_char * 4)
    ]

class pcal_struct(Mk4StructureBase):
    _fields_ = [
        ('chan_id', ctypes.c_char * 8),
        ('frequency', ctypes.c_float ),
        ('real', ctypes.c_float ),
        ('imaginary', ctypes.c_float )
    ]

class type_308(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('time', date),
        ('duration', ctypes.c_float),
        ('pcal', pcal_struct * 32)
    ]

class ch1_tag(Mk4StructureBase):
    _fields_ = [
        ('chan_name', ctypes.c_char * 8),
        ('freq', ctypes.c_double ),
        ('acc', (ctypes.c_uint * 2) * 64 )
    ]

class type_309(Mk4StructureBase):
    _fields_ = [
        ('record_id', ctypes.c_char * 3),
        ('version_no', ctypes.c_char * 2),
        ('unused1', ctypes.c_char * 3),
        ('su', ctypes.c_int),
        ('ntones', ctypes.c_int),
        ('rot', ctypes.c_double),
        ('acc_period', ctypes.c_double),
        ('chan', ch1_tag * 64)
    ]

class model_struct(Mk4StructureBase):
    _fields_ = [
        ('chan_id', ctypes.c_char * 32),
        ('t301', ctypes.POINTER(type_301) * 64), #MAXSPLINES
        ('t302', ctypes.POINTER(type_302) * 64),
        ('t303', ctypes.POINTER(type_303) * 64),
    ]

class mk4_sdata(Mk4StructureBase):
    _fields_ = [
        ('allocated', ctypes.POINTER(None) * (2*64*64 + 7*3600 + 3)), #2*MAXSPLINES*MAXFREQ + 7*MAXSTATPER + 3
        ('nalloc', ctypes.c_int),
        ('file_image', ctypes.POINTER(ctypes.c_char)),
        ('id', ctypes.POINTER(type_000)),
        ('t300', ctypes.POINTER(type_300)),
        ('model', model_struct * 64), #MAXFREQ
        ('n304', ctypes.c_int),
        ('n305', ctypes.c_int),
        ('n306', ctypes.c_int),
        ('n307', ctypes.c_int),
        ('n308', ctypes.c_int),
        ('n309', ctypes.c_int),
        ('t304', ctypes.POINTER(type_304) * 3600), #MAXSTATPER
        ('t305', ctypes.POINTER(type_305) * 3600),
        ('t306', ctypes.POINTER(type_306) * 3600),
        ('t307', ctypes.POINTER(type_307) * 3600),
        ('t308', ctypes.POINTER(type_308) * 3600),
        ('t309', ctypes.POINTER(type_309) * 3600)
    ]

    def __del__(self):
        """clean up memory allocated by c library"""
        dfio = mk4io_load()
        dfio.clear_mk4sdata(ctypes.byref(self))

################################################################################
# library calls
################################################################################

def mk4io_load():
    prefix = os.getenv('HOPS_PREFIX')
    root = os.getenv('HOPS_ROOT')
    arch = os.getenv('HOPS_ARCH')
    if prefix != None:
        path = '/'.join([prefix,'lib','hops','libmk4io.so'])
        dfio = ctypes.cdll.LoadLibrary(path)
        return dfio
    elif (root is None) or (arch is None):
        #hops env not set up yet, need to find the library using LD_LIBRARY_PATH
        ld_lib_path = os.getenv('LD_LIBRARY_PATH')
        if ld_lib_path is None:
            return None #we failed
        possible_path_list = ld_lib_path.split(':')
        for a_path in possible_path_list:
            libpath = '/'.join([a_path,'libmk4io.so'])
            if os.path.isfile(libpath):
                #found the library, go ahead and load it up
                dfio = ctypes.cdll.LoadLibrary(libpath)
                return dfio
        #otherwise we didn't find the library
        return None
    else:
        path = '/'.join([root,arch,'lib','hops','libmk4io.so'])
        dfio = ctypes.cdll.LoadLibrary(path)
        return dfio

def mk4corel(filename):
    """read and return mk4corel file object"""
    mk4p = mk4_corel()
    dfio = mk4io_load()
    dfio.read_mk4corel(ctypes.c_char_p(filename.encode()), ctypes.byref(mk4p))
    return mk4p

def mk4fringe(filename):
    """read and return a mk4fringe file object"""
    mk4p = mk4_fringe()
    dfio = mk4io_load()
    dfio.read_mk4fringe(ctypes.c_char_p(filename.encode()), ctypes.byref(mk4p))
    return mk4p

def clear_mk4fringe(mk4_fringe_obj):
    """clear/delete a mk4fringe object"""
    dfio = mk4io_load()
    dfio.clear_mk4fringe(ctypes.byref(mk4_fringe_obj))

def mk4sdata(filename):
    """read an return a mk4sdata file object"""
    mk4p = mk4_sdata()
    dfio = mk4io_load()
    dfio.read_mk4sdata(ctypes.c_char_p(filename.encode()), ctypes.byref(mk4p))
    return mk4p

# eof
