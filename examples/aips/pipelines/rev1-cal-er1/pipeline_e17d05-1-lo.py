'''
This is a sample ParselTongue pipeline script for EHT 2017 data, which is yet
under development. Here I describe the overall path for reducing e17b06-1-lo
band data sets.

This pipeline should work for other datasets, because I don't use any
specific antenna IDs and source names in the script.

Developer: Kazu Akiyama
Ver: 2017/10/16
'''
#-------------------------------------------------------------------------------
# Load Modules
#-------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

#-------------------------------------------------------------------------------
# Pipeline Parameters
#   *** Please edit following parameters ***
#-------------------------------------------------------------------------------
# AIPS Parameters
#   AIPS_ROOT Directory
#   (if you already loaded LOGIN.SH, you don't have to set this)
aipsdir="/usr/local/aips"
#   AIPS USER NO
userno=11
#   AIPS DISK
disk=1

# ParselTongue Parameters
#   ParselTongue Python Directory
ptdir="/usr/share/parseltongue/python"
#   Obit Python Directory
obitdir="/usr/lib/obit/python"

# Observational Parameters
obscode="e17d05"
rev=1
band="lo"
ver=1

# IF range for fringe-fitting
# if bif=1 at low band, we will miss JCMT in KRING-fitting.
bif = 2
eif = 32

# Workdir where output files will be saved.
workdir='./%s-%d-%s-ver%d'%(obscode,rev,band,ver)
fileheader="%s-%d%s"%(obscode,rev,band[0])

# FITS Files
#   Here FITS files are assumed to be unpacked with unpack_rev1_fitsidi.py
#   in eat/exampls/aips/scripts.
#
#   FITS Dirctories for High and Low bands
fitsdir='/xxxxx/rev1-fitsidi/%s-%d-%s'%(obscode,rev,band)

# AIPS AN table correction file
ancortab=os.path.join("ancortab_2017apr.csv")


#-------------------------------------------------------------------------------
# Load eat.aips modules
#-------------------------------------------------------------------------------
# Set envrioments
import eat.aips as ea

# Use AIPS 31DEC17
ea.setenv(aipsdir=aipsdir, aipsver="31DEC17", ptdir=ptdir, obitdir=obitdir)

# Load parseltoungue and eat.aips functions
from eat.aips.aipsutil import *
from eat.aips.ehtutil  import *

#-------------------------------------------------------------------------------
# Initialize AIPS and AIPS parameters
#-------------------------------------------------------------------------------
if os.path.isdir(workdir) is False:
    os.system("mkdir -p %s"%(workdir))

#-------------------------------------------------------------------------------
# Initialize AIPS and AIPS parameters
#-------------------------------------------------------------------------------
setuser(userno)   # userid
tv=AIPSTV() # AIPSTV

# AIPS UV Data
uvname="%s-%d%s"%(obscode,rev,band[0])
uvname=uvname.upper()
uvdata=AIPSUVData(uvname, 'FITLD', disk, 1)
msortdata=AIPSUVData(uvdata.name, 'MSORT', disk, 1)
tasav1data=AIPSUVData(uvdata.name, 'TASAV1', disk, 1)
splat1data=AIPSUVData(uvdata.name, 'SPLAT1', disk, 1)
fixwt1data=AIPSUVData(uvdata.name, 'FIXWT1', disk, 1)
tasav2data=AIPSUVData(uvdata.name, 'TASAV2', disk, 1)
splat2data=AIPSUVData(uvdata.name, 'SPLAT2', disk, 1)
fixwt2data=AIPSUVData(uvdata.name, 'FIXWT2', disk, 1)


#-------------------------------------------------------------------------------
# Data Loading, Sorting, Indexing
#-------------------------------------------------------------------------------
# Use AIPS VER 31DEC16 tentatively
ea.setenv(aipsdir=aipsdir, aipsver="31DEC16", ptdir=ptdir, obitdir=obitdir)


# Make fits loader
loaderdir=os.path.split(fitsdir)[0]
loaderdir=os.path.join(loaderdir, "aips-loader-%s-%d-%s"%(obscode,rev,band))
refdate=mkfitsloader(fitsdir, loaderdir, filename="loader.fits", skipna=True)

# A quick short cut to run fitld
ehtload(
    outdata=uvdata,
    datain=os.path.join(loaderdir,"loader.fits"),
    refdate=refdate,
    clint=1/60.)

# Use AIPS VER 31DEC17 tentatively
ea.setenv(aipsdir=aipsdir, aipsver="31DEC17", ptdir=ptdir, obitdir=obitdir)

# Data sorting and indexing (MSORT / INDXR)
ehtsort(
    indata=uvdata,
    outdata=msortdata,
    clint=1/60.)

#-------------------------------------------------------------------------------
# GET some info useful in later procedures
#-------------------------------------------------------------------------------
# Number of Antennas
Nant=len(msortdata.antennas)
Nch=msortdata.header["naxis"][2]
Nif=msortdata.header["naxis"][3]

# Check if SR and SM exists
isSR = "SR" in msortdata.antennas
isSM = "SM" in msortdata.antennas

# Station Code
AAid=msortdata.antennaids("AA")
if isSR:
    SRid=msortdata.antennaids("SR")
if isSM:
    SMid=msortdata.antennaids("SM")

# Non ALMA station IDs
NoAAs=msortdata.antennas
NoAAs.remove("AA")
NoAAs=msortdata.antennaids(NoAAs)
NoAAs.insert(0, None)


#-------------------------------------------------------------------------------
# Data Summary
#-------------------------------------------------------------------------------
fileid=1

# This runs PRTAN, LISTR, DTSUM at once
ehtsumm(
    indata=msortdata,
    prtanout=os.path.join(workdir,"%s.%02d.prtan.txt"%(fileheader,fileid)),
    listrout=os.path.join(workdir,"%s.%02d.listr.scan.txt"%(fileheader,fileid+1)),
    dtsumout=os.path.join(workdir,"%s.%02d.dtsum.txt"%(fileheader,fileid+2)),
    overwrite=True)
fileid+= 3


#-------------------------------------------------------------------------------
# Correct Reciever Type in AIPS Data
#-------------------------------------------------------------------------------
# Correct Reciever Mount Type
#    This runs PRTAN, LISTR, DTSUM at once
ehtancor(indata=msortdata, datain=ancortab)


# PRTAN
ehtsumm(
    indata=msortdata,
    prtanout=os.path.join(workdir,"%s.%02d.ancor.prtan.txt"%(fileheader,fileid)),
    overwrite=True)
fileid+= 3

#-------------------------------------------------------------------------------
# Check Autocorrelation Data
#-------------------------------------------------------------------------------
# Time vs AC Power: Whole Observation
ehtvplotac(
    indata=msortdata,
    pltifs=10,
    outfile=os.path.join(workdir,"%s.%02d.vplot.ac.if10.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# Time vs AC Power: Scan-basis
ehtvplotac(
    indata=msortdata,
    pltifs=10,
    tmode="eachscan",
    outfile=os.path.join(workdir,"%s.%02d.vplot.ac.if10.eachscan.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# Frequency vs (Scan-averaged) AC Power: Whole IFs
ehtpossmac(
    msortdata,
    bindif=True,
    bindpol=True,
    outfile=os.path.join(workdir,"%s.%02d.possm.ac.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


#-------------------------------------------------------------------------------
# Flag first and last 5 channels at each IF
#-------------------------------------------------------------------------------
# UVFLG
task=tget("uvflg")
task.getn(msortdata)
task.outfgver=-1
task.opcode="FLAG"
task.reason="IF EDGE"
task.dohist=1
task.bchan=1
task.echan=5
task()
task.bchan=Nch-4
task.echan=Nch
task()

# Save to files (TBOUT)
task=tget("tbout")
task.getn(msortdata)
task.inext='FG'
task.inver=msortdata.table_highver("FG")
task.outtext=os.path.join(workdir,"%s.%02d.fg.ifedge.tbout"%(fileheader,fileid))
task.docrt=-1
task.check(overwrite=True)
task()
fileid += 1

# Save to files (CSV FILE)
fgtab=ehtfgout(msortdata)
fgtab.to_csv(os.path.join(workdir,"%s.%02d.fg.ifedge.csv"%(fileheader,fileid)))
fileid += 1

#-------------------------------------------------------------------------------
# First ACSCL
#-------------------------------------------------------------------------------
# 1st ACSCL
ehtacscl(
    msortdata,
    solint=1e-6,
    gainver=msortdata.table_highver("CL"),
    gainuse=msortdata.table_highver("CL")+1,
    doband=5,
    bpver=-1,
    flagmode=2 # 1: run SNEDT, 2: run EDITA and flag Data based on SN table
)

# Save to files
task=tget("tbout")
task.getn(msortdata)
task.inext='FG'
task.inver=msortdata.table_highver("FG")
task.outtext=os.path.join(workdir,"%s.%02d.fg.acscl2.tbout"%(fileheader,fileid))
task.docrt=-1
task.check(overwrite=True)
task()
fileid += 1

fgtab=ehtfgout(msortdata)
fgtab.to_csv(os.path.join(workdir,"%s.%02d.fg.acscl2.csv"%(fileheader,fileid)))
fileid += 1

# Plot SN Table
ehtsnplt(
    indata=msortdata,
    pltifs=10,
    inext="SN",
    invers=msortdata.table_highver("SN"),
    optypes="AMP",
    outfile=os.path.join(workdir,"%s.%02d.acscl.sn.if10.ps"%(fileheader,fileid)),
    nplots=Nant,
    overwrite=True,
    zappltabs=True)
fileid += 1

# Plot CL Table
ehtsnplt(
    indata=msortdata,
    pltifs=10,
    inext="CL",
    optypes="AMP",
    invers=msortdata.table_highver("CL"),
    outfile=os.path.join(workdir,"%s.%02d.acscl.cl.if10.ps"%(fileheader,fileid)),
    nplots=Nant,
    overwrite=True,
    zappltabs=True)
fileid += 1

# Time vs AC Power: Whole Observation
ehtvplotac(
    indata=msortdata,
    pltifs=10,
    docalib=1,
    gainuse=msortdata.table_highver("CL"),
    outfile=os.path.join(workdir,"%s.%02d.acscl.vplot.ac.if10.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# ------------------------------------------------------------------------------
# BPASS (tentative) & FLAGGING CHANNELS
#     Here, I tentatively solve AC bandpass, and flag channels with BPEDT
# ------------------------------------------------------------------------------
# BPASS
ehtbpassac(
    msortdata,
    docal=1,
    gainuse=msortdata.table_highver("CL"),
    solint=-1)

# BPEDT
tv.restart()
task=tget("bpedt")
task.getn(msortdata)
task.outfgver=msortdata.table_highver("FG")+1
task.reason="BAD AC CHANNEL"
task.dohist=1
task()
# This is correcting a bug in BPEDT that all FLAGS of BPEDT has
# a source code and subarray of 1
ehtfgreset(msortdata)

# Save to files
task=tget("tbout")
task.getn(msortdata)
task.inext='FG'
task.inver=msortdata.table_highver("FG")
task.outtext=os.path.join(workdir,"%s.%02d.fg.bpedt.tbout"%(fileheader,fileid))
task.docrt=-1
task.check(overwrite=True)
task()
fileid += 1

fgtab=ehtfgout(msortdata)
fgtab.to_csv(os.path.join(workdir,"%s.%02d.fg.bpedt.csv"%(fileheader,fileid)))
fileid += 1

# Zap tempral BP table
msortdata.zap_table("BP", 1)

#-------------------------------------------------------------------------------
# AC Bandpass
#-------------------------------------------------------------------------------
# BPASS Solve Bandpass on every 1 minute
tables=msortdata.tables
ehtbpassac(
    msortdata,
    docal=1,
    gainuse=msortdata.table_highver("CL"),
    solint=1.)

# Frequency vs (Scan-averaged) BP table: Whole IFs
ehtpossmac(
    msortdata,
    bindif=True,
    plotbp=True,
    docalib=1,
    gainuse=msortdata.table_highver("CL"),
    doband=1,
    bpver=1,
    solint=10,
    outfile=os.path.join(workdir,"%s.%02d.bpassac.possm.bp.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# Frequency vs (Scan-averaged) AC Spectrum: Whole IFs
ehtpossmac(
    msortdata,
    bindif=True,
    plotbp=False,
    docalib=1,
    gainuse=msortdata.table_highver("CL"),
    doband=5,
    bpver=1,
    outfile=os.path.join(workdir,"%s.%02d.bpassac.possm.ac.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# Time vs AC Power: Whole Observation
ehtvplotac(
    indata=msortdata,
    pltifs=10,
    docalib=1,
    gainuse=msortdata.table_highver("CL"),
    doband=5,
    bpver=1,
    outfile=os.path.join(workdir,"%s.%02d.bpassac.vplot.ac.if10.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

#-------------------------------------------------------------------------------
# Second ACSCL: Correct small drifts in AC power induced by BPASS
#-------------------------------------------------------------------------------
# 2nd ACSCL
tables=msortdata.tables
ehtacscl(
    msortdata,
    solint=1e-6,
    gainver=msortdata.table_highver("CL"),
    gainuse=msortdata.table_highver("CL")+1,
    doband=5,
    bpver=1,
    flagmode=0 # Just for checking
)

# Save to files
task=tget("tbout")
task.getn(msortdata)
task.inext='FG'
task.inver=msortdata.table_highver("FG")
task.outtext=os.path.join(workdir,"%s.%02d.fg.acscl2.tbout"%(fileheader,fileid))
task.docrt=-1
task.check(overwrite=True)
task()
fileid += 1

fgtab=ehtfgout(msortdata)
fgtab.to_csv(os.path.join(workdir,"%s.%02d.fg.acscl2.csv"%(fileheader,fileid)))
fileid += 1

# Plot SN Table
ehtsnplt(
    indata=msortdata,
    pltifs=10,
    inext="SN",
    invers=msortdata.table_highver("SN"),
    optypes="AMP",
    outfile=os.path.join(workdir,"%s.%02d.acscl2.sn.if10.ps"%(fileheader,fileid)),
    nplots=Nant,
    overwrite=True,
    zappltabs=True)
fileid += 1

# Plot CL Table
ehtsnplt(
    indata=msortdata,
    pltifs=10,
    inext="CL",
    invers=msortdata.table_highver("CL"),
    optypes="AMP",
    outfile=os.path.join(workdir,"%s.%02d.acscl2.cl.if10.ps"%(fileheader,fileid)),
    nplots=Nant,
    overwrite=True,
    zappltabs=True)
fileid += 1

# Time vs AC Power: Whole Observation
ehtvplotac(
    indata=msortdata,
    pltifs=10,
    docalib=1,
    gainuse=msortdata.table_highver("CL"),
    doband=5,
    bpver=1,
    outfile=os.path.join(workdir,"%s.%02d.acscl2.vplot.ac.if10.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# Frequency vs (Scan-averaged) AC Spectrum: Whole IFs
ehtpossmac(
    msortdata,
    bindif=True,
    plotbp=False,
    docalib=1,
    gainuse=msortdata.table_highver("CL"),
    doband=5,
    bpver=1,
    outfile=os.path.join(workdir,"%s.%02d.acscl2.possm.ac.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

#-------------------------------------------------------------------------------
# Save Solutions and SPLAT Data
#-------------------------------------------------------------------------------
# TASAV
zap(tasav1data)
task=tget("tasav")
task.getn(msortdata)
task.geton(tasav1data)
task()

task=tget("fittp")
task.getn(tasav1data)
task.dataout=os.path.join(workdir,"%s.%02d.tasav1.fittp"%(fileheader,fileid))
task.check(overwrite=True)
task()# TASAV
fileid += 1

# SPLAT: applying solutions so far, and also flagging edge IFs
zap(splat1data)
task=tget("splat")
task.getn(msortdata)
task.geton(splat1data)
task.bif=bif
task.eif=eif
task.docalib=1
task.gainuse=0
task.doband=5
task.bpver=1
task()
Nif-=1

#-------------------------------------------------------------------------------
# Recalculate weights on each visibility:
#    This helps later KRING runs to get more reliable solutions, since KRING
#    analytically estimates SNRs.
#-------------------------------------------------------------------------------
zap(fixwt1data)
task=tget("fixwt")
task.getn(splat1data)
task.geton(fixwt1data)
task.solint=3
task()

task=tget("indxr")
task.getn(fixwt1data)
task()


#-------------------------------------------------------------------------------
# Parallactic Angle Correction
#-------------------------------------------------------------------------------
# Parallactic Angle Correction
ehtpang(fixwt1data)

# Plot CL Table
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="CL",
    invers=fixwt1data.table_highver("CL"),
    outfile=os.path.join(workdir,"%s.%02d.pang.cl.if10.ps"%(fileheader,fileid)),
    nplots=Nant,
    optypes="PHAS",
    overwrite=True,
    zappltabs=True)
fileid += 1

# Plot CC Spectra
ehtpossmcc(
    indata=fixwt1data,
    docalib=1,
    gainuse=fixwt1data.table_highver("CL"),
    doband=-1,
    bpver=-1,
    solint=-1,
    bindif=True,
    plotbp=False,
    outfile=os.path.join(workdir,"%s.%02d.pang.cc.ps"%(fileheader,fileid)),
    pagetype=0,
    overwrite=True,
    zappltabs=True)
fileid += 1


#-------------------------------------------------------------------------------
# UVFLG: make SMAP/SMAR independent
#     SR (SMA reference) is a station of the phased SMA staion (SM).
#     So, SM and SR should have a lot of correlating noises, and should be
#     flagged in global-fringe searching process.
#-------------------------------------------------------------------------------
if isSR and isSM:
    task=tget("uvflg")
    task.getn(fixwt1data)
    task.antennas[1]=int(np.min([SRid,SMid]))
    task.baseline[1]=int(np.max([SRid,SMid]))
    task.outfgver=fixwt1data.table_highver("FG")+1
    task.opcode="FLAG"
    task.reason="FLAG SMAP/SMAR BL"
    task.dohist=1
    task()

    task=tget("tbout")
    task.getn(fixwt1data)
    task.inext='FG'
    task.inver=fixwt1data.table_highver("FG")
    task.outtext=os.path.join(workdir,"%s.%02d.fg.sma.tbout"%(fileheader,fileid))
    task.docrt=-1
    task.check(overwrite=True)
    task()
    fileid += 1

    fgtab=ehtfgout(fixwt1data)
    fgtab.to_csv(os.path.join(workdir,"%s.%02d.fg.sma.csv"%(fileheader,fileid)))
    fileid += 1


#-------------------------------------------------------------------------------
# First Fringe Search:
#    Here, I start from solving RATE, MBD, PHASE on scan basis (NO SBD!!).
#    I realized that solutions in KRING can be improved by running FIXWT
#    before KRING. This helps, for instance, L-R delay differences more stable.
#-------------------------------------------------------------------------------
tables=fixwt1data.tables
task=tget("kring")
task.getn(fixwt1data)
task.docalib=1
task.gainuse=fixwt1data.table_highver("CL")
task.refant=AAid
task.search[1]=AAid
task.solint=10.     # solint in minutes
task.solmode="RD"   # N: solve for each IF, R: RATE, D: MBD, S: averaged SBD
task.doifs=1        # if 'N' not in solmode, specify how many solutions at each segment
task.opcode=""      # ZRAT: reset rates, ZPHA: reset phases: ZDEL: reset delays
task.cparm[1]=1     # Data integration time: 1sec (see DTSUM)
task.cparm[2]=220   # Delay window (cf. 59 MHz * 106 nsec ~ 2*pi)
task.cparm[3]=0     # Rate window (cf. 100 mHz * 10sec ~ 1 rotation)
task.cparm[4]=5     # SNR Cutoff
task.cparm[5]=3     # Number of baseline combinations
task.cparm[6]=1     # 0 means exhaustive search (aparm(9)=1 in AIPS)
                    # 1 means ignoring non-ALMA data
task.prtlev=1
task()


# Plot SN Table (colored by IF/polarizations)
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    antennas=NoAAs,
    optypes=["PHAS","RATE","DELA","SNR"],
    nplots=Nant-1,
    do3col=1,
    outfile=os.path.join(workdir,"%s.%02d.kring1.sn.pol.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# Plot SN Table (colored by sources)
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    antennas=NoAAs,
    optypes=["PHAS","RATE","DELA","SNR"],
    nplots=Nant-1,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring1.sn.src.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# Plot SN Table (RL diffrences, colored by IF/polarizations)
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    stokes="DIFF",
    antennas=NoAAs,
    optypes=["PHAS","RATE","DELA"],
    nplots=Nant-3,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring1.sn.diff.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# CLCAL
task=tget("clcal")
task.getn(fixwt1data)
task.interpol="SELF"
task.cutoff=10
task.snver=fixwt1data.table_highver("SN")
task.gainver=fixwt1data.table_highver("CL")
task.gainuse=fixwt1data.table_highver("CL")+1
task.refant=AAid
task()

# PLOT CL TABLE
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="CL",
    invers=fixwt1data.table_highver("CL"),
    outfile=os.path.join(workdir,"%s.%02d.kring1.cl.if10.ps"%(fileheader,fileid)),
    optypes=["PHAS", "DELA", "RATE"],
    nplots=Nant,
    overwrite=True,
    zappltabs=True)
fileid += 1

ehtpossmcc(
    indata=fixwt1data,
    docalib=1,
    gainuse=fixwt1data.table_highver("CL"),
    doband=-1,
    bpver=-1,
    solint=-1,
    bindif=True,
    plotbp=False,
    outfile=os.path.join(workdir,"%s.%02d.kring1.cc.ps"%(fileheader,fileid)),
    pagetype=0,
    overwrite=True,
    zappltabs=True)
fileid += 1

#-------------------------------------------------------------------------------
# Second Fringe Search:
#    Here, I solved residual instrumental phase/delay offsets on scan basis
#    You may realize that I did not flag any solutions in the previous stage.
#    This is because KRING sometimes misfits delays, even data have enough SNRs.
#    Such miss fit will be corrected in this stage.
#
#    This search may be better to be done after third search, bu I'm still
#    figuring out the way to do this.
#-------------------------------------------------------------------------------
# KRING
tables=fixwt1data.tables
task=tget("kring")
task.getn(fixwt1data)
task.docalib=1
task.gainuse=fixwt1data.table_highver("CL")
task.refant=AAid
task.search[1]=AAid
task.solint=10.     # solint in minutes
task.solmode="ND"   # N: solve for each IF, R: RATE, D: MBD, S: averaged SBD
task.doifs=Nif      # if 'N' not in solmode, specify how many solutions at each segment
task.opcode=""      # ZRAT: reset rates, ZPHA: reset phases: ZDEL: reset delays
task.cparm[1]=1     # Data integration time: 1sec (see DTSUM)
task.cparm[2]=220   # Delay window (cf. 59 MHz * 106 nsec ~ 2*pi)
task.cparm[3]=0     # Rate window (cf. 100 mHz * 10sec ~ 1 rotation)
task.cparm[4]=5     # SNR Cutoff
task.cparm[5]=3     # Number of baseline combinations
task.cparm[6]=1     # 0 means exhaustive search (aparm(9)=1 in AIPS)
                    # 1 means ignoring non-ALMA data
task.prtlev=1
task()


# Plot SN Table
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    antennas=NoAAs,
    optypes=["PHAS","DELA","SNR"],
    nplots=Nant-1,
    do3col=1,
    outfile=os.path.join(workdir,"%s.%02d.kring2.sn.pol.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


# Plot SN Table
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    antennas=NoAAs,
    optypes=["PHAS","DELA","SNR"],
    nplots=Nant-1,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring2.sn.src.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    stokes="DIFF",
    antennas=NoAAs,
    optypes=["PHAS","DELA"],
    nplots=Nant-3,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring2.sn.diff.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


# CLCAL
task=tget("clcal")
task.getn(fixwt1data)
task.interpol="SELF"
task.cutoff=10
task.snver=fixwt1data.table_highver("SN")
task.gainver=fixwt1data.table_highver("CL")
task.gainuse=fixwt1data.table_highver("CL")+1
task.refant=AAid
task()


# PLOT CL TABLE
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="CL",
    invers=fixwt1data.table_highver("CL"),
    outfile=os.path.join(workdir,"%s.%02d.kring2.cl.if10.ps"%(fileheader,fileid)),
    optypes=["PHAS", "DELA", "RATE"],
    nplots=Nant,
    overwrite=True,
    zappltabs=True)
fileid += 1

ehtpossmcc(
    indata=fixwt1data,
    docalib=1,
    gainuse=fixwt1data.table_highver("CL"),
    doband=-1,
    bpver=-1,
    solint=-1,
    bindif=True,
    plotbp=False,
    outfile=os.path.join(workdir,"%s.%02d.kring2.cc.ps"%(fileheader,fileid)),
    pagetype=0,
    overwrite=True,
    zappltabs=True)
fileid += 1

#-------------------------------------------------------------------------------
# Third Fringe Search:
#    Here, I solve phase rotations. IFs are binded again to maximize SNRs.
#-------------------------------------------------------------------------------
# KRING
tables=fixwt1data.tables
task=tget("kring")
task.getn(fixwt1data)
task.docalib=1
task.gainuse=fixwt1data.table_highver("CL")
task.refant=AAid
task.search[1]=AAid
task.solint=2./60.  # solint in minutes
task.solmode="R"    # N: solve for each IF, R: RATE, D: MBD, S: averaged SBD
task.doifs=1        # if 'N' not in solmode, specify how many solutions at each segment
task.opcode=""      # ZRAT: reset rates, ZPHA: reset phases: ZDEL: reset delays
task.cparm[1]=1     # Data integration time: 1sec (see DTSUM)
task.cparm[2]=0     # Delay window (cf. 59 MHz * 106 nsec ~ 2*pi)
task.cparm[3]=0     # Rate window (cf. 100 mHz * 10sec ~ 1 rotation)
task.cparm[4]=5     # SNR Cutoff
task.cparm[5]=3     # Number of baseline combinations
task.cparm[6]=1     # 0 means exhaustive search (aparm(9)=1 in AIPS)
                    # 1 means ignoring non-ALMA data
task.prtlev=1
task()


# Plot SN Table
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    antennas=NoAAs,
    optypes=["PHAS","RATE","SNR"],
    nplots=Nant-1,
    do3col=1,
    outfile=os.path.join(workdir,"%s.%02d.kring3.sn.pol.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


# Plot SN Table
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    antennas=NoAAs,
    optypes=["PHAS","RATE","SNR"],
    nplots=Nant-1,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring3.sn.src.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    stokes="DIFF",
    antennas=NoAAs,
    optypes=["PHAS","RATE"],
    nplots=Nant-3,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring3.sn.diff.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    tmode="eachscan",
    antennas=NoAAs,
    optypes=["PHAS","RATE"],
    nplots=Nant-1,
    do3col=1,
    outfile=os.path.join(workdir,"%s.%02d.kring3.sn.eachscan.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    tmode="eachscan",
    stokes="DIFF",
    antennas=NoAAs,
    optypes=["PHAS","RATE"],
    nplots=Nant-3,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring3.sn.diff.eachscan.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1

# CLCAL
task=tget("clcal")
task.getn(fixwt1data)
task.interpol="SELF"
task.cutoff=10
task.snver=fixwt1data.table_highver("SN")
task.gainver=fixwt1data.table_highver("CL")
task.gainuse=fixwt1data.table_highver("CL")+1
task.refant=AAid
task()

# PLOT CL TABLE
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="CL",
    invers=fixwt1data.table_highver("CL"),
    outfile=os.path.join(workdir,"%s.%02d.kring3.cl.if10.ps"%(fileheader,fileid)),
    optypes=["PHAS", "DELA", "RATE"],
    nplots=Nant,
    overwrite=True,
    zappltabs=True)
fileid += 1

ehtpossmcc(
    indata=fixwt1data,
    docalib=1,
    gainuse=fixwt1data.table_highver("CL"),
    doband=-1,
    bpver=-1,
    solint=-1,
    bindif=True,
    plotbp=False,
    outfile=os.path.join(workdir,"%s.%02d.kring3.cc.ps"%(fileheader,fileid)),
    pagetype=0,
    overwrite=True,
    zappltabs=True)
fileid += 1

#-------------------------------------------------------------------------------
# Fourth Fringe Search:
#    Here, I solved residual instrumental phase/delay/rate offsets on scan basis.
#-------------------------------------------------------------------------------
# KRING
tables=fixwt1data.tables
task=tget("kring")
task.getn(fixwt1data)
task.docalib=1
task.gainuse=fixwt1data.table_highver("CL")
task.refant=AAid
task.search[1]=AAid
task.solint=10.     # solint in minutes
task.solmode="NDR"  # N: solve for each IF, R: RATE, D: MBD, S: averaged SBD
task.doifs=Nif      # if 'N' not in solmode, specify how many solutions at each segment
task.opcode=""      # ZRAT: reset rates, ZPHA: reset phases: ZDEL: reset delays
task.cparm[1]=1     # Data integration time: 1sec (see DTSUM)
task.cparm[2]=50    # Delay window (cf. 59 MHz * 106 nsec ~ 2*pi)
task.cparm[3]=50    # Rate window (cf. 100 mHz * 10sec ~ 1 rotation)
task.cparm[4]=5     # SNR Cutoff
task.cparm[5]=3     # Number of baseline combinations
task.cparm[6]=1     # 0 means exhaustive search (aparm(9)=1 in AIPS)
                    # 1 means ignoring non-ALMA data
task.prtlev=1
task()


# Plot SN Table
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    antennas=NoAAs,
    optypes=["PHAS","DELA","SNR"],
    nplots=Nant-1,
    do3col=1,
    outfile=os.path.join(workdir,"%s.%02d.kring4.sn.pol.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


# Plot SN Table
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    antennas=NoAAs,
    optypes=["PHAS","DELA","SNR"],
    nplots=Nant-1,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring4.sn.src.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="SN",
    invers=fixwt1data.table_highver("SN"),
    stokes="DIFF",
    antennas=NoAAs,
    optypes=["PHAS","DELA"],
    nplots=Nant-3,
    do3col=2,
    outfile=os.path.join(workdir,"%s.%02d.kring4.sn.diff.ps"%(fileheader,fileid)),
    overwrite=True,
    zappltabs=True)
fileid += 1


# CLCAL
task=tget("clcal")
task.getn(fixwt1data)
task.interpol="SELF"
task.cutoff=10
task.snver=fixwt1data.table_highver("SN")
task.gainver=fixwt1data.table_highver("CL")
task.gainuse=fixwt1data.table_highver("CL")+1
task.refant=AAid
task()


# PLOT CL TABLE
ehtsnplt(
    indata=fixwt1data,
    pltifs=10,
    inext="CL",
    invers=fixwt1data.table_highver("CL"),
    outfile=os.path.join(workdir,"%s.%02d.kring4.cl.if10.ps"%(fileheader,fileid)),
    optypes=["PHAS", "DELA", "RATE"],
    nplots=Nant,
    overwrite=True,
    zappltabs=True)
fileid += 1


ehtpossmcc(
    indata=fixwt1data,
    docalib=1,
    gainuse=fixwt1data.table_highver("CL"),
    doband=-1,
    bpver=-1,
    solint=-1,
    bindif=True,
    plotbp=False,
    outfile=os.path.join(workdir,"%s.%02d.kring4.cc.ps"%(fileheader,fileid)),
    pagetype=0,
    overwrite=True,
    zappltabs=True)
fileid += 1

#-------------------------------------------------------------------------------
# Complex Bandpass:
#-------------------------------------------------------------------------------
ehtbpasscc(
    indata=fixwt1data,
    docal=1,
    gainuse=fixwt1data.table_highver("CL"),
    doband=-1,
    bpver=-1,
    solint=3)

ehtpossmcc(
    indata=fixwt1data,
    docalib=1,
    gainuse=fixwt1data.table_highver("CL"),
    doband=5,
    bpver=fixwt1data.table_highver("BP"),
    solint=-1,
    bindif=True,
    plotbp=True,
    outfile=os.path.join(workdir,"%s.%02d.bpass.bp.ps"%(fileheader,fileid)),
    pagetype=0,
    overwrite=True,
    zappltabs=True)
fileid += 1

ehtpossmcc(
    indata=fixwt1data,
    docalib=1,
    gainuse=fixwt1data.table_highver("CL"),
    doband=5,
    bpver=fixwt1data.table_highver("BP"),
    solint=-1,
    bindif=True,
    plotbp=False,
    outfile=os.path.join(workdir,"%s.%02d.bpass.cc.ps"%(fileheader,fileid)),
    pagetype=0,
    overwrite=True,
    zappltabs=True)
fileid += 1


#-------------------------------------------------------------------------------
# Save Solutions and SPLAT Data
#-------------------------------------------------------------------------------
# TASAV
zap(tasav2data)
task=tget("tasav")
task.getn(fixwt1data)
task.geton(tasav2data)
task()

task=tget("fittp")
task.getn(tasav2data)
task.dataout=os.path.join(workdir,"%s.%02d.tasav2.fittp"%(fileheader,fileid))
task.check(overwrite=True)
task()# TASAV
fileid += 1

# SPLAT: applying solutions so far, and also flagging edge IFs
zap(splat2data)
task=tget("splat")
task.getn(fixwt1data)
task.geton(splat2data)
task.bif=1
task.eif=Nif
task.docalib=1
task.gainuse=0
task.aparm[1]=1
task.doband=5
task.bpver=1
task()

#-------------------------------------------------------------------------------
# Recalculate weights on each visibility:
#-------------------------------------------------------------------------------
zap(fixwt2data)
task=tget("fixwt")
task.getn(splat2data)
task.geton(fixwt2data)
task.solint=3
task()

task=tget("indxr")
task.getn(fixwt2data)
task()

task=tget("fittp")
task.getn(fixwt2data)
task.dataout=os.path.join(workdir,"%s.%02d.fixwt2.fittp"%(fileheader,fileid))
task.check(overwrite=True)
task()# TASAV
fileid += 1

#-------------------------------------------------------------------------------
# SPLIT DATA
#-------------------------------------------------------------------------------
sources = fixwt2data.sources

for source in sources:
    splitdata = AIPSUVData(source, "SPLIT", uvdata.disk, 1)
    zap(splitdata)

task = tget("split")
task.getn(fixwt2data)
task.outclass="SPLIT"
task()

for source in sources:
    splitdata = AIPSUVData(source, "SPLIT", uvdata.disk, 1)
    task=tget("fittp")
    task.getn(splitdata)
    task.dataout=os.path.join(workdir,"%s.%02d.%s.fittp"%(fileheader,fileid,source))
    task.check(overwrite=True)
    task()# TASAV
    fileid += 1
