'''
This is a sample ParselTongue pipeline script for EHT 2017 data, which is yet 
under development. 

Developer: Kazu Akiyama
Ver: 2017/09/11
'''
import eat.aips as ea
ea.set_env()
#
import os
#
from AIPS import AIPS, AIPSDisk
from AIPSData import AIPSUVData, AIPSImage
import eat.aips.aipstask as aipstask 
import eat.aips.ehtutil as ehtutil

workdir = 'XXXX/eht2017/e17b06-1-lo-rev1'
fitsdir = 'XXXX/eht2017data/fitsidi-rev1/aipsloader-e17b06-1-lo'
fitsname = "LOADER.FITS"
ancortab = os.path.join(workdir, "ancortab_2017apr.csv")
userno = 2
disk=1
uvname = "E17B06-1L"


#-------------------------------------------------------------------------------
# Initialize AIPS Parameters
#-------------------------------------------------------------------------------
AIPS.userno = userno
uvdata = AIPSUVData(uvname, 'FITLD', disk, 1)
msortdata = AIPSUVData(uvdata.name, 'MSORT', uvdata.disk, 1)


#-------------------------------------------------------------------------------
# Data Loading, Sorting, Indexing
#-------------------------------------------------------------------------------
ehtutil.ehtload(
    outdata=uvdata,
    datain=os.path.join(fitsdir,"LOADER.FITS"),
    clint=1/60.)


# Data sorting and indexing
ehtutil.ehtsort(
    indata=uvdata,
    outdata=msortdata,
    clint=1/60.)


#-------------------------------------------------------------------------------
# Data Summary
#-------------------------------------------------------------------------------
# Data Summary
fileid = 1
ehtutil.ehtsumm(
    indata=msortdata,
    docrt=-1,
    prtanout=os.path.join(workdir,"e17b06-1l.%02d.prtan.txt"%(fileid)),
    listrout=os.path.join(workdir,"e17b06-1l.%02d.listr.scan.txt"%(fileid+1)),
    dtsumout=os.path.join(workdir,"e17b06-1l.%02d.dtsum.txt"%(fileid+2)),
    overwrite=True)
fileid+= 3


#-------------------------------------------------------------------------------
# Correct Reciever Type in AIPS Data
#-------------------------------------------------------------------------------
# Correct Reciever Mount Type
ehtutil.ehtancor(
    indata=msortdata,
    datain=ancortab)

task = aipstask.AIPSTask("prtan")
task.defaults()
task.getn(msortdata)
task.docrt=-1
task.outprint=os.path.join(workdir,"e17b06-1l.%02d.prtan.tabed.txt"%(fileid))
task.check(overwrite=True)
task()
fileid += 1

