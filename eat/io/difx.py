from ctypes import *

difxio = cdll.LoadLibrary('/home/lindy/difx/lib/libdifxio.so')

class DifxAntennaFlag(Structure):
    _fields_ = [
    ('mjd1', c_double),
    ('mjd2', c_double),
    ('antennaId', c_int),
]

class DifxJob(Structure):
    _fields_ = [
    ('difxVersion', c_char * 64),
    ('difxLabel', c_char * 64),
    ('jobStart', c_double),
    ('jobStop', c_double),
    ('mjdStart', c_double),
    ('duration', c_double),
    ('jobId', c_int),
    ('subjobId', c_int),
    ('subarrayId', c_int),
    ('obsCode', c_char * 8),
    ('obsSession', c_char * 8),
    ('taperFunction', c_int),
    ('calcServer', c_char * 32),
    ('calcVersion', c_int),
    ('calcProgram', c_int),
    ('activeDatastreams', c_int),
    ('activeBaselines', c_int),
    ('polyOrder', c_int),
    ('polyInterval', c_int),
    ('aberCorr', c_int),
    ('dutyCycle', c_double),
    ('nFlag', c_int),
    ('flag', POINTER(DifxAntennaFlag)),
    ('vexFile', c_char * 256),
    ('inputFile', c_char * 256),
    ('calcFile', c_char * 256),
    ('imFile', c_char * 256),
    ('flagFile', c_char * 256),
    ('threadsFile', c_char * 256),
    ('outputFile', c_char * 256),
    ('jobIdRemap', POINTER(c_int)),
    ('freqIdRemap', POINTER(c_int)),
    ('antennaIdRemap', POINTER(c_int)),
    ('datastreamIdRemap', POINTER(c_int)),
    ('baselineIdRemap', POINTER(c_int)),
    ('pulsarIdRemap', POINTER(c_int)),
    ('configIdRemap', POINTER(c_int)),
    ('sourceIdRemap', POINTER(c_int)),
    ('spacecraftIdRemap', POINTER(c_int)),
]

class DifxIF(Structure):
    _fields_ = [
    ('freq', c_double),
    ('bw', c_double),
    ('sideband', c_char),
    ('nPol', c_int),
    ('pol', c_char * 2),
    ('rxName', c_char * 8),
]

class DifxConfig(Structure):
    _fields_ = [
    ('name', c_char * 32),
    ('tInt', c_double),
    ('subintNS', c_int),
    ('guardNS', c_int),
    ('fringeRotOrder', c_int),
    ('strideLength', c_int),
    ('xmacLength', c_int),
    ('numBufferedFFTs', c_int),
    ('pulsarId', c_int),
    ('phasedArrayId', c_int),
    ('nPol', c_int),
    ('pol', c_char * 2),
    ('polMask', c_int),
    ('doPolar', c_int),
    ('doAutoCorr', c_int),
    ('quantBits', c_int),
    ('nAntenna', c_int),
    ('nDatastream', c_int),
    ('nBaseline', c_int),
    ('datastreamId', POINTER(c_int)),
    ('baselineId', POINTER(c_int)),
    ('nIF', c_int),
    ('IF', POINTER(DifxIF)),
    ('fitsFreqId', c_int),
    ('freqId2IF', POINTER(c_int)),
    ('freqIdUsed', POINTER(c_int)),
    ('ant2dsId', POINTER(c_int)),
]

class DifxStringArray(Structure):
    _fields_ = [
    ('str', POINTER(POINTER(c_char))),
    ('n', c_int),
    ('nAlloc', c_int),
]

class DifxRule(Structure):
    _fileds_ = [
    ('sourceName', DifxStringArray),
    ('scanId', DifxStringArray),
    ('calCode', c_char * 4),
    ('qual', c_int),
    ('mjdStart', c_double),
    ('mjdStop', c_double),
    ('configName', c_char * 32),
]

class DifxFreq(Structure):
    _fields_ = [
    ('freq', c_double),
    ('bw', c_double),
    ('sideband', c_char),
    ('nChan', c_int),
    ('specAvg', c_int),
    ('overSamp', c_int),
    ('decimation', c_int),
    ('nTone', c_int),
    ('tone', POINTER(c_int)),
    ('rxName', c_char * 8),
]

class DifxAntenna(Structure):
    _fields_ = [
    ('name', c_char * 32),
    ('origId', c_int),
    ('clockrefmjd', c_double),
    ('clockorder', c_int),
    ('clockcoeff', c_double * (5 + 1)),
    ('mount', c_int),
    ('siteType', c_int),
    ('offset', c_double * 3),
    ('X', c_double),
    ('Y', c_double),
    ('Z', c_double),
    ('dX', c_double),
    ('dY', c_double),
    ('dZ', c_double),
    ('spacecraftId', c_int),
    ('shelf', c_char * 8),
]

class DifxPolyModel(Structure):
    _fields_ = [
    ('mjd', c_int),
    ('sec', c_int),
    ('order', c_int),
    ('validDuration', c_int),
    ('delay', c_double * (5 + 1)),
    ('dry', c_double * (5 + 1)),
    ('wet', c_double * (5 + 1)),
    ('az', c_double * (5 + 1)),
    ('elcorr', c_double * (5 + 1)),
    ('elgeom', c_double * (5 + 1)),
    ('parangle', c_double * (5 + 1)),
    ('u', c_double * (5 + 1)),
    ('v', c_double * (5 + 1)),
    ('w', c_double * (5 + 1)),
]

class DifxPolyModelExtension(Structure):
    _fields_ = [
    ('delta', c_double),
    ('dDelay_dl', c_double * (5 + 1)),
    ('dDelay_dm', c_double * (5 + 1)),
    ('d2Delay_dldl', c_double * (5 + 1)),
    ('d2Delay_dldm', c_double * (5 + 1)),
    ('d2Delay_dmdm', c_double * (5 + 1)),
]

class DifxPolyModelLMExtension(Structure):
    _fields_ = [
    ('delta', c_double),
    ('dDelay_dl', c_double * (5 + 1)),
    ('dDelay_dm', c_double * (5 + 1)),
    ('d2Delay_dldl', c_double * (5 + 1)),
    ('d2Delay_dldm', c_double * (5 + 1)),
    ('d2Delay_dmdm', c_double * (5 + 1)),
]

class DifxPolyModelXYZExtension(Structure):
    _fields_ = [
    ('delta', c_double),
    ('dDelay_dX', c_double * (5 + 1)),
    ('dDelay_dY', c_double * (5 + 1)),
    ('dDelay_dZ', c_double * (5 + 1)),
    ('d2Delay_dXdX', c_double * (5 + 1)),
    ('d2Delay_dXdY', c_double * (5 + 1)),
    ('d2Delay_dXdZ', c_double * (5 + 1)),
    ('d2Delay_dYdY', c_double * (5 + 1)),
    ('d2Delay_dYdZ', c_double * (5 + 1)),
    ('d2Delay_dZdZ', c_double * (5 + 1)),
]

class DifxScan(Structure):
    _fields_ = [
    ('mjdStart', c_double),
    ('mjdEnd', c_double),
    ('startSeconds', c_int),
    ('durSeconds', c_int),
    ('identifier', c_char * 32),
    ('obsModeName', c_char * 32),
    ('maxNSBetweenUVShifts', c_int),
    ('maxNSBetweenACAvg', c_int),
    ('pointingCentreSrc', c_int),
    ('nPhaseCentres', c_int),
    ('phsCentreSrcs', c_int * 1000),
    ('orgjobPhsCentreSrcs', c_int * 1000),
    ('jobId', c_int),
    ('configId', c_int),
    ('nAntenna', c_int),
    ('nPoly', c_int),
    ('im', POINTER(POINTER(POINTER(DifxPolyModel)))),
    ('imLM', POINTER(POINTER(POINTER(DifxPolyModelLMExtension)))),
    ('imXYZ', POINTER(POINTER(POINTER(DifxPolyModelXYZExtension)))),
]

class DifxSource(Structure):
    _fields_ = [
    ('ra', c_double),
    ('dec', c_double),
    ('name', c_char * 32),
    ('calCode', c_char * 4),
    ('qual', c_int),
    ('spacecraftId', c_int),
    ('numFitsSourceIds', c_int),
    ('fitsSourceIds', POINTER(c_int)),
    ('pmRA', c_double),
    ('pmDec', c_double),
    ('parallax', c_double),
    ('pmEpoch', c_double),
]

class DifxEOP(Structure):
    _fields_ = [
    ('mjd', c_int),
    ('tai_utc', c_int),
    ('ut1_utc', c_double),
    ('xPole', c_double),
    ('yPole', c_double),
]

class DifxDatastream(Structure):
    _fields_ = [
    ('antennaId', c_int),
    ('tSys', c_float),
    ('dataFormat', c_char * 128),
    ('dataSampling', c_int),
    ('nFile', c_int),
    ('file', POINTER(POINTER(c_char))),
    ('networkPort', c_char * 12),
    ('windowSize', c_int),
    ('quantBits', c_int),
    ('dataFrameSize', c_int),
    ('dataSource', c_int),
    ('phaseCalIntervalMHz', c_int),
    ('tcalFrequency', c_int),
    ('nRecTone', c_int),
    ('recToneFreq', POINTER(c_int)),
    ('recToneOut', POINTER(c_int)),
    ('clockOffset', POINTER(c_double)),
    ('clockOffsetDelta', POINTER(c_double)),
    ('phaseOffset', POINTER(c_double)),
    ('freqOffset', POINTER(c_double)),
    ('nRecFreq', c_int),
    ('nRecBand', c_int),
    ('nRecPol', POINTER(c_int)),
    ('recFreqId', POINTER(c_int)),
    ('recBandFreqId', POINTER(c_int)),
    ('recBandPolName', POINTER(c_char)),
    ('nZoomFreq', c_int),
    ('nZoomBand', c_int),
    ('nZoomPol', POINTER(c_int)),
    ('zoomFreqId', POINTER(c_int)),
    ('zoomBandFreqId', POINTER(c_int)),
    ('zoomBandPolName', POINTER(c_char)),
]

class DifxBaseline(Structure):
    _fields_ = [
    ('dsA', c_int),
    ('dsB', c_int),
    ('nFreq', c_int),
    ('nPolProd', POINTER(c_int)),
    ('bandA', POINTER(POINTER(c_int))),
    ('bandB', POINTER(POINTER(c_int))),
]

class sixVector(Structure):
    _fields_ = [
    ('mjd', c_int),
    ('fracDay', c_double),
    ('X', c_longdouble),
    ('Y', c_longdouble),
    ('Z', c_longdouble),
    ('dX', c_longdouble),
    ('dY', c_longdouble),
    ('dZ', c_longdouble),
]

class RadioastronTimeFrameOffset(Structure):
    _fields_ = [
    ('Delta_t', c_double),
    ('dtdtau', c_double),
]

class RadioastronAxisVectors(Structure):
    _fields_ = [
    ('X', c_double * 3),
    ('Y', c_double * 3),
    ('Z', c_double * 3),
]

class DifxSpacecraft(Structure):
    _fields_ = [
    ('name', c_char * 32),
    ('nPoint', c_int),
    ('pos', POINTER(sixVector)),
    ('timeFrameOffset', POINTER(RadioastronTimeFrameOffset)),
    ('axisVectors', POINTER(RadioastronAxisVectors)),
    ('frame', c_char * 32),
]

class DifxPolyco(Structure):
    _fields_ = [
    ('fileName', c_char * 256),
    ('dm', c_double),
    ('refFreq', c_double),
    ('mjd', c_double),
    ('nCoef', c_int),
    ('nBlk', c_int),
    ('p0', c_double),
    ('f0', c_double),
    ('coef', POINTER(c_double)),
]

class DifxPulsar(Structure):
    _fields_ = [
    ('fileName', c_char * 256),
    ('nPolyco', c_int),
    ('polyco', POINTER(DifxPolyco)),
    ('nBin', c_int),
    ('binEnd', POINTER(c_double)),
    ('binWeight', POINTER(c_double)),
    ('scrunch', c_int),
]

class DifxPhasedArray(Structure):
    _fields_ = [
    ('fileName', c_char * 256),
    ('outputType', c_int),
    ('outputFormat', c_int),
    ('accTime', c_double),
    ('complexOutput', c_int),
    ('quantBits', c_int),
]

class DifxInput(Structure):
    _fields_ = [
    ('fracSecondStartTime', c_int),
    ('mjdStart', c_double),
    ('mjdStop', c_double),
    ('refFreq', c_double),
    ('startChan', c_int),
    ('specAvg', c_int),
    ('nInChan', c_int),
    ('nOutChan', c_int),
    ('visBufferLength', c_int),
    ('nIF', c_int),
    ('nPol', c_int),
    ('doPolar', c_int),
    ('nPolar', c_int),
    ('chanBW', c_double),
    ('quantBits', c_int),
    ('polPair', c_char * 4),
    ('dataBufferFactor', c_int),
    ('nDataSegments', c_int),
    ('outputFormat', c_int),
    ('eopMergeMode', c_int),
    ('nCore', c_int),
    ('nThread', POINTER(c_int)),
    ('nAntenna', c_int),
    ('nConfig', c_int),
    ('nRule', c_int),
    ('nFreq', c_int),
    ('nScan', c_int),
    ('nSource', c_int),
    ('nEOP', c_int),
    ('nFlag', c_int),
    ('nDatastream', c_int),
    ('nBaseline', c_int),
    ('nSpacecraft', c_int),
    ('nPulsar', c_int),
    ('nPhasedArray', c_int),
    ('nJob', c_int),
    ('job', POINTER(DifxJob)),
    ('config', POINTER(DifxConfig)),
    ('rule', POINTER(DifxRule)),
    ('freq', POINTER(DifxFreq)),
    ('antenna', POINTER(DifxAntenna)),
    ('scan', POINTER(DifxScan)),
    ('source', POINTER(DifxSource)),
    ('eop', POINTER(DifxEOP)),
    ('datastream', POINTER(DifxDatastream)),
    ('baseline', POINTER(DifxBaseline)),
    ('spacecraft', POINTER(DifxSpacecraft)),
    ('pulsar', POINTER(DifxPulsar)),
    ('phasedarray', POINTER(DifxPhasedArray)),
]

def loadDifxInput(filePrefix = '/media/lindy/DATA/smm-pc/d21us_clean/h_1609'):
    difxio.loadDifxInput.restype = POINTER(DifxInput)
    a = difxio.loadDifxInput(c_char_p(filePrefix))
    return a.contents
