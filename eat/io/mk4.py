from ctypes import *

mk4io = cdll.LoadLibrary('/home/lindy/hops-3.10/sub/mk4.so')

# int
# read_mk4corel (char *filename,
#                struct mk4_corel *corel)

class type_000(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('date', c_char * 16),
    ('name', c_char * 40),
]

class type_100(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('date', c_char * 16),
    ('name', c_char * 40),
]

class date(Structure):
    _fields_ = [
    ('year', c_short),
    ('day', c_short),
    ('hour', c_short),
    ('minute', c_short),
    ('second', c_float),
]

class type_100(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('procdate', date),
    ('baseline', c_char * 2),
    ('rootname', c_char * 34),
    ('qcode', c_char * 2),
    ('unused2', c_char * 6),
    ('pct_done', c_float),
    ('start', date),
    ('stop', date),
    ('ndrec', c_int),
    ('nindex', c_int),
    ('nlags', c_short),
    ('nblocks', c_short),
]

class type_101(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('status', c_char),
    ('nblocks', c_short),
    ('index', c_short),
    ('primary', c_short),
    ('ref_chan_id', c_char * 8),
    ('rem_chan_id', c_char * 8),
    ('corr_board', c_short),
    ('corr_slot', c_short),
    ('ref_chan', c_short),
    ('rem_chan', c_short),
    ('post_mortem', c_int),
    ('blocks', c_int),
]

class flag_wgt(Union):
    _fields_ = [
    ('flag', c_int),
    ('weight', c_float),
]

class counts_per_lag(Structure):
    _fields_ = [
    ('coscor', c_int),
    ('cosbits', c_int),
    ('sincor', c_int),
    ('sinbits', c_int),
]

class lag_tag(Structure):
    _fields_ = [
    ('coscor', c_int),
    ('sincor', c_int),
]

class counts_global(Structure):
    _fields_ = [
    ('cosbits', c_int),
    ('sinbits', c_int),
    ('lags', lag_tag),
]

class auto_per_lag(Structure):
    _fields_ = [
    ('coscor', c_int),
    ('cosbits', c_int),
]

class auto_global(Structure):
    _fields_ = [
    ('cosbits', c_int),
    ('unused', c_char * 4),
    ('coscor', c_int),
]

class spectral(Structure):
    _fields_ = [
    ('re', c_float),
    ('im', c_float),
]

class lag_data(Union):
    _fields_ = [
    ('cpl', counts_per_lag),
    ('cg', counts_global),
    ('apl', auto_per_lag),
    ('ag', auto_global),
    ('spec', spectral),
]

class type_120(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('type', c_char),
    ('nlags', c_short),
    ('baseline', c_char * 2),
    ('rootcode', c_char * 6),
    ('index', c_int),
    ('ap', c_int),
    ('fw', flag_wgt),
    ('status', c_int),
    ('fr_delay', c_int),
    ('delay_rate', c_int),
    ('ld', lag_data),
]

class index_tag(Structure):
    _fields_ = [
    ('t101', POINTER(type_101)),   # a single t101 record
    ('ap_space', c_int),
    ('t120', POINTER(POINTER(type_120))),
]

class mk4_corel(Structure):
    _fields_ = [
    ('allocated', POINTER(None) * (8192 + 4)), # pointers to allocated blocks for memory management
    ('nalloc', c_int),                         # the number of such pointers
    ('file_image', POINTER(c_char)),
    ('id', POINTER(type_000)),                 # single record
    ('t100', POINTER(type_100)),               # single record
    ('index_space', c_int),                    # length of index array
    ('index', POINTER(index_tag)),
]

def loadmk4(filePrefix = '/media/lindy/DATA/dewi/3424/080-0500_LOW/FD..wxfrmh'):
    # difxio.loadDifxInput.restype = POINTER(DifxInput)
    mk4p = mk4_corel()
    mk4io.read_mk4corel(c_char_p(filePrefix), byref(mk4p))
    return mk4p

class type_300(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 2),
    ('SU_number', c_ubyte),
    ('id', c_char),
    ('intl_id', c_char * 2),
    ('name', c_char * 32),
    ('unused2', c_char),
    ('model_start', date),
    ('model_interval', c_float),
    ('nsplines', c_short),
]

class type_301(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('interval', c_short),
    ('chan_id', c_char * 32),
    ('unused2', c_char * 6),
    ('delay_spline', c_double * 6),
]

class type_302(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('interval', c_short),
    ('chan_id', c_char * 32),
    ('unused2', c_char * 6),
    ('phase_spline', c_double * 6),
]

class type_303(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('interval', c_short),
    ('chan_id', c_char * 32),
    ('unused2', c_char * 6),
    ('azimuth', c_double * 6),
    ('elevation', c_double * 6),
    ('parallactic_angle', c_double * 6),
    ('u', c_double * 6),
    ('v', c_double * 6),
    ('w', c_double * 6),
]

class type_304_anon(Structure):
    _fields_ = [
    ('error_count', c_int),
    ('frames', c_int),
    ('bad_frames', c_int),
    ('slip_sync', c_int),
    ('missing_sync', c_int),
    ('crc_error', c_int),
]

class type_304(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('time', date),
    ('duration', c_float),
    ('trackstats', type_304_anon * 64),
]

class type_305(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
]

class type_306_anon(Structure):
    _fields_ = [
    ('chan_id', c_char * 32),
    ('bigpos', c_int),
    ('pos', c_int),
    ('neg', c_int),
    ('bigneg', c_int),
]


class type_306(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('time', date),
    ('duration', c_float),
    ('stcount', type_306_anon * 16),
]

class ChanCount(Structure):
    _fields_ = [
    ('count', c_uint32 * 8),
    ('val_count', c_uint32),
]

class type_307(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('su', c_int),
    ('unused2', c_char * 4),
    ('tot', c_double),
    ('rot', c_double),
    ('accum_period', c_double),
    ('frame_count', c_uint32),
    ('counts', ChanCount * 16),
    ('unused3', c_char * 4),
]

# /home/lindy/hops-3.10/include/type_308.h: 17
class type_308_anon(Structure):
    _fields_ = [
    ('chan_id', c_char * 8),
    ('frequency', c_float),
    ('real', c_float),
    ('imaginary', c_float),
]

class type_308(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('time', date),
    ('duration', c_float),
    ('pcal', type_308_anon * 32),
]

class ch1_tag(Structure):
    _fields_ = [
    ('chan_name', c_char * 8),
    ('freq', c_double),
    ('acc', (c_uint32 * 2) * 64),
]

class type_309(Structure):
    _fields_ = [
    ('record_id', c_char * 3),
    ('version_no', c_char * 2),
    ('unused1', c_char * 3),
    ('su', c_int),
    ('ntones', c_int),
    ('rot', c_double),
    ('acc_period', c_double),
    ('chan', ch1_tag * 16),
]

class mk4_sdata_anon(Structure):
    _fields_ = [
    ('chan_id', c_char * 32),
    ('t301', POINTER(type_301) * 64),
    ('t302', POINTER(type_302) * 64),
    ('t303', POINTER(type_303) * 64),
]

# /home/lindy/hops-3.10/include/mk4_data.h: 111
class mk4_sdata(Structure):
    _fields_ = [
    ('allocated', POINTER(None) * ((((2 * 64) * 64) + (7 * 3600)) + 3)),
    ('nalloc', c_int),
    ('file_image', POINTER(c_char_p)),
    ('id', POINTER(type_000)),
    ('t300', POINTER(type_300)),
    ('model', mk4_sdata_anon * 64),
    ('n304', c_int),
    ('n305', c_int),
    ('n306', c_int),
    ('n307', c_int),
    ('n308', c_int),
    ('n309', c_int),
    ('t304', POINTER(type_304) * 3600),
    ('t305', POINTER(type_305) * 3600),
    ('t306', POINTER(type_306) * 3600),
    ('t307', POINTER(type_307) * 3600),
    ('t308', POINTER(type_308) * 3600),
    ('t309', POINTER(type_309) * 3600),
]

def loadsdata(filePrefix):
    mk4p = mk4_sdata()
    mk4io.read_mk4sdata(c_char_p(filePrefix), byref(mk4p))
    return mk4p

class m4view:
    def __init__(mk4p):
        self.mk4p = mk4p



