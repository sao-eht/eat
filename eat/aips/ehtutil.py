#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A python module eat.aips.aipstask

This is a submodule of eat.aips. This submodule contains EHT-specific functions
for AIPS reduction using ParselTongue.

This module must be loaded after rnnning eat.aips.set_env().
'''
# ------------------------------------------------------------------------------
# Loading Modules
# ------------------------------------------------------------------------------
# Check if ParselTongue modeules can be load.
import eat.aips as ea
ea.check(printver=False)
from eat.aips.aipsutil import *
#
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# Flag Tables and Data Flagging
# ------------------------------------------------------------------------------
fgtab_columns = "ant1,ant2,bchan,echan,freqid,bif,eif".split(",")
fgtab_columns+= "stokesRR,stokesLL,stokesRL,stokesLR".split(",")
fgtab_columns+= "source,subarray,timerange1,timerange2,reason".split(",")


def ehtfgout(indata, fgver=0):
    if fgver == 0:
        fgv = indata.table_highver("FG")
    else:
        fgv = fgver

    # Get data
    fgtab = indata.table("FG", fgv)
    ants = indata.antennas
    srcs = indata.sources
    ants.insert(0,"ALL")
    srcs.insert(0,"ALL")

    # Make a dictionary
    outdata = {}
    for key in fgtab_columns:
        outdata[key] = []

    # Read data from the input FG table
    Ntab = len(fgtab)
    for itab in np.arange(Ntab):
        cont = fgtab[itab]
        outdata["ant1"].append(ants[cont["ants"][0]])
        outdata["ant2"].append(ants[cont["ants"][1]])
        outdata["bchan"].append(cont["chans"][0])
        outdata["echan"].append(cont["chans"][1])
        outdata["bif"].append(cont["ifs"][0])
        outdata["eif"].append(cont["ifs"][1])
        outdata["freqid"].append(cont["freq_id"])
        outdata["stokesRR"].append(bool(cont["pflags"][0]))
        outdata["stokesLL"].append(bool(cont["pflags"][1]))
        outdata["stokesRL"].append(bool(cont["pflags"][2]))
        outdata["stokesLR"].append(bool(cont["pflags"][3]))
        outdata["reason"].append(cont["reason"])
        outdata["source"].append(srcs[cont["source"]])
        outdata["subarray"].append(cont["subarray"])
        timerange1 = fday2timerang(cont["time_range"][0])
        timerange2 = fday2timerang(cont["time_range"][1])
        outdata["timerange1"].append(
            "%+04d:%02d:%02d:%02d"%(timerange1[0],timerange1[1],
                                    timerange1[2],timerange1[3]))
        outdata["timerange2"].append(
            "%+04d:%02d:%02d:%02d"%(timerange2[0],timerange2[1],
                                    timerange2[2],timerange2[3]))
    outdata = pd.DataFrame(outdata, columns=fgtab_columns)
    return outdata


def ehtfgin(indata, fgtab, outfgver=-1, unflag=False):
    # Correct information from indata
    ants = indata.antennas
    srcs = indata.sources
    ants.insert(0,"ALL")
    srcs.insert(0,"ALL")
    Nant = len(ants)

    # check fgtab
    for key in fgtab_columns:
        fgtab[key]
    Ntab = fgtab.shape[0]

    for itab in np.arange(Ntab):
        tab = fgtab.loc[itab, :]
        task = tget("uvflg")
        task.getn(indata)
        # source
        if tab["source"].upper() != "ALL":
            task.sources[1] = tab.loc[0,"source"].upper()
        # freqid
        task.freqid = int(tab["freqid"])
        # timerang
        timerange1 = np.asarray(tab["timerange1"].split(":"), dtype=np.int64).tolist()
        timerange2 = np.asarray(tab["timerange2"].split(":"), dtype=np.int64).tolist()
        timerang = [None] + timerange1 + timerange2
        task.timerang = timerang
        # bchan, echan, bif, eif
        task.bchan = int(tab["bchan"])
        task.echan = int(tab["echan"])
        task.bif = int(tab["bif"])
        task.eif = int(tab["eif"])
        task.outfgver = outfgver

        # antennas
        ant1 = tab["ant1"].upper()
        ant2 = tab["ant2"].upper()
        if ant1 not in ants:
            print("Antenna %s is not in uvdata"%(ant1))
            continue
        else:
            for iant1 in np.arange(Nant):
                if ant1 == ants[iant1]:
                    break
        if ant2 not in ants:
            print("Antenna %s is not in uvdata"%(ant2))
            continue
        else:
            for iant2 in np.arange(Nant):
                if ant2 == ants[iant2]:
                    break
        task.antennas[1] = int(iant1)
        task.baseline[1] = int(iant2)
        # stokes
        isRR, isLL, isRL, isLR = tab[["stokesRR", "stokesLL",
                                             "stokesRL", "stokesLR"]]
        if isRR & isLL & isRL & isLR:
            task.stokes="FULL"
        elif (not isRR) & (not isLL) & isRL & isLR:
            task.stokes="CROS"
        elif isRR & isLL & (not isRL) & (not isLR):
            task.stokes="HALF"
        elif isRR & (not isLL) & isRL & isLR:
            task.stokes="RR"
        elif (not isRR) & isLL & isRL & isLR:
            task.stokes="LL"
        elif (not isRR) & (not isLL) & isRL & (not isLR):
            task.stokes="RL"
        elif (not isRR) & (not isLL) & (not isRL) & isLR:
            task.stokes="LR"
        else:
            comment = "Cannnot Handle input Stokes flags:"
            comment+= "RR: %r, "%(isRR)
            comment+= "LL: %r, "%(isLL)
            comment+= "RL: %r, "%(isRL)
            comment+= "LR: %r"%(isLR)
        # outfgver
        task.outfgver = -1
        if not unflag:
            task.opcode="FLAG"
        else:
            task.opcode="UNFLAG"
        task.reason=tab["reason"][0:24]
        task.dohist=1
        task()


def ehtfgreset(indata, fgver=0):
    if fgver==0:
        fgver=indata.table_highver("FG")

    fgtab=indata.table("FG", fgver)
    Ntab = len(fgtab)
    for itab in np.arange(Ntab):
        tab = fgtab[itab]
        if tab["source"]!=0:
            task = tget("tabed")
            task.getn(indata)
            task.inext = "FG"
            task.optype = "REPL"
            task.inver = fgver
            task.outver = fgver
            task.aparm[1] = 1
            task.aparm[4] = 4
            task.bcount = int(itab+1)
            task.ecount = int(itab+1)
            task.keyvalue[1] = 0
            task()
        if tab["subarray"]!=0:
            task = tget("tabed")
            task.getn(indata)
            task.inext = "FG"
            task.optype = "REPL"
            task.inver = fgver
            task.outver = fgver
            task.aparm[1] = 2
            task.aparm[4] = 4
            task.bcount = int(itab+1)
            task.ecount = int(itab+1)
            task.keyvalue[1] = 0
            task()


# ------------------------------------------------------------------------------
# Print Out Summary
# ------------------------------------------------------------------------------
def ehtsumm(indata, prtanout=None, listrout=None, dtsumout=None, overwrite=False):
    # PRTAN
    if prtanout is not None:
        task = tget("prtan")
        task.getn(indata)
        task.docrt=-1
        task.outprint=prtanout
        task.check(overwrite=overwrite)
        task()


    # LISTR (SCAN)
    if listrout is not None:
        task = tget("listr")
        task.getn(indata)
        task.optype='SCAN'
        task.docrt=-1
        task.outprint=listrout
        task.check(overwrite=overwrite)
        task()


    # DTSUM
    if dtsumout is not None:
        task = tget("dtsum")
        task.getn(indata)
        task.aparm[1]=2
        task.docrt=-1
        task.outprint=dtsumout
        task.check(overwrite=overwrite)
        task()

# ------------------------------------------------------------------------------
# Data Loading and Sorting
# ------------------------------------------------------------------------------
def mkfitsloader(fitsdir, outdir, filename="loader.fits", skipna=True, skip31if=True):
    import astropy.io.fits as pf
    import astropy.time as at
    from tqdm import tqdm
        
    # Check data in FITS files
    datetimes = []
    refdates = []
    fitsnames = []
    list1 = os.listdir(fitsdir)
    for comp in tqdm(list1, bar_format="Reading FITS directory: "+r'{l_bar}{bar}{r_bar}'):
        comppath=os.path.join(fitsdir,comp)
        if "na-" in comp and skipna:
            continue
        if not os.path.isfile(comppath):
            continue
        try:
            hdulist = pf.open(comppath)
        except IOError:
            continue
        
        
        # Get UV Data
        uvdata = hdulist["UV_DATA"]
        
        # Check number of IFs
        if uvdata.header["MAXIS4"]!=32:
            continue
        
        # FITS Files
        fitsnames.append(comppath)
        
        # Get Time Stamp
        times = at.Time(uvdata.data["DATE"], format="jd", scale="utc")
        times+= at.TimeDelta(uvdata.data["TIME"], format="jd")
        hdulist.close()
        datetimes.append(times.min().datetime)
    
    fitsfiles = {'datetime':datetimes,'fitsfile':fitsnames}
    fitsfiles = pd.DataFrame(fitsfiles, columns=["datetime", "fitsfile"])
    fitsfiles = fitsfiles.sort_values(by="datetime").reset_index(drop=True)
    Nfile = len(fitsfiles.fitsfile)
    print("  - %d FITS files are found"%(Nfile))
    print(fitsfiles)
    
    os.system("mkdir -p %s"%(outdir))
    os.system("rm -rf %s*"%(os.path.join(outdir,filename)))
    for i in tqdm(xrange(Nfile), bar_format="Creating symbolic links: "+r'{l_bar}{bar}{r_bar}'):
        orgfile = os.path.relpath(fitsfiles.loc[i, "fitsfile"], start=outdir)
        lnfile = "%s%d"%(filename,i+1)
        os.system("cd %s; ln -s %s %s"%(outdir, orgfile, lnfile))
    
    refdate = fitsfiles.loc[0, "datetime"]
    refdate = "%04d%2d%02d"%(refdate.year, refdate.month, refdate.day)
    return refdate

def ehtload(
        outdata,
        datain="",
        ncount=1000,
        refdate="",
        clint=1/60.):
    '''
    Load FITS-IDI files into AIPS using FITLD.

    Args:
      outdata (AIPSUVData object):
        output AIPSUVData

      datain (str):
        FITS Filename

      ncount (int; default=1000):
        The number of input FITS files. (see FITLD HELP for details)

      clint (int; default=1/60.):
        Interval for CL tables.
    '''
    import astropy.io.fits as pf
        
    zap(outdata)
    task = tget("fitld")
    task.geton(outdata)
    task.datain=datain
    task.ncount=ncount
    task.doconcat=1
    task.clint=clint
    task.refdate=refdate
    task.check()
    task()


def ehtsort(indata, outdata, clint=1/60.):
    '''
    Sort and Indexing UVDATA using MSORT and INDXR

    Args:
      indata (AIPSUVData object):
        input AIPSUVData

      outdata (AIPSUVData object):
        output AIPSUVData

      datain (str):
        FITS Filename

      clint (int; default=1/60.):
        Interval for CL tables.
    '''
    zap(outdata)
    task = tget("msort")
    task.getn(indata)
    task.geton(outdata)
    task()

    task = tget("indxr")
    task.getn(outdata)
    task.cparm[3] = clint
    task()


# ------------------------------------------------------------------------------
# Antab Correction
# ------------------------------------------------------------------------------
def ehtancor(indata, inver=0, datain=""):
    '''
    Corrects mislabled reciever-mount types of EHT stations in AN table

    Args:
      indata (AIPSUVData object):
        input data

      inver (int):
        Version of the AN table to be corrected. This function overwrites this
        input AN table.

      datain (str):
        Filename for a csv correction table.
    '''
    # Get correction Table
    cortable = pd.read_csv(datain)
    print(cortable)
    annames_tab = list(set(cortable["ANNAME"]))

    # Get Antennna information
    annames = indata.antennas
    Nan = len(annames)

    # Correct a specified AIPS AN Table with TABED
    for ian in range(Nan):
        anname = annames[ian]
        if anname not in annames_tab:
            print("[WARNING] No correction info for the station %s"%(anname))
            continue
        mntsta = cortable.loc[cortable["ANNAME"]==anname,"MNTSTA"].reset_index(drop=True)[0]
        #
        task = tget("tabed")
        task.getn(indata)
        task.inext = "AN"
        task.optype = "REPL"
        task.inver = inver
        task.outver = -1
        task.aparm[1] = 5
        task.aparm[4] = 4
        task.bcount = int(ian+1)
        task.ecount = int(ian+1)
        task.keyvalue[1] = int(mntsta)
        #print(ian+1,anname,mntsta)
        task()


# ------------------------------------------------------------------------------
# Parallactic Angle Correction
# ------------------------------------------------------------------------------
def ehtpang(indata):
    '''
    Corrects phases for parallactic angle effects.

    Args:
      indata (AIPSUVData object):
        input data
    '''
    # get old tables
    oldtabs = indata.tables

    # LISTR (SCAN)
    task = tget("clcor")
    task.getn(indata)
    task.gainver = indata.table_highver("CL")
    task.gainuse = task.gainver + 1
    task.opcode = "PANG"
    task.clcorprm[1] = 1
    task()

    # Show updated tables
    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtpang: new tables created")
    print(diftables)
    return diftables


# ------------------------------------------------------------------------------
# ACCOR
# ------------------------------------------------------------------------------
def ehtaccor(
        indata,
        solint=1./120.,
        calsour=[None,""],
        sources=[None,""],
        flagmode=0):
    '''
    '''

    if flagmode > 0.5:
        tv=AIPSTV()
        tv.restart()

    # get number of IFs
    Nch = indata.header["naxis"][2]
    Nif = indata.header["naxis"][3]

    # get old tables
    oldtabs = indata.tables

    # Run ACSCL
    task = tget("accor")
    task.getn(indata)
    task.solint=solint
    task()
    SNver = indata.table_highver("SN")

    if flagmode==1:
        # Flagging data
        task = tget("snedt")
        task.getn(indata)
        task.inext="SN"
        task.invers=SNver
        task()
        SNver = indata.table_highver("SN")
    elif flagmode==2:
        # Flagging data
        task = tget("edita")
        task.getn(indata)
        task.inext="SN"
        task.invers=SNver
        task.outfgver=indata.table_highver("FG")+1
        task.reason="BAD ACCOR SOLUTION"
        task()
        ehtfgreset(indata, fgver=0)

    # CLCAL
    task = tget("clcal")
    task.getn(indata)
    task.sources=sources
    task.calsour=calsour
    task.interpol="2PT"
    task.snver = SNver
    task.gainver = indata.table_highver("CL")
    task.gainuse = indata.table_highver("CL")+1
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtaccor: new tables created")
    print(diftables)
    return diftables


def ehtacscl(
        indata,
        solint=1./120.,
        calsour=[None,""],
        sources=[None,""],
        gainver=0,
        gainuse=0,
        doband=0,
        bpver=-1,
        flagmode=0):
    '''
    '''
    if flagmode > 0.5:
        tv=AIPSTV()
        if not tv.exists():
            tv.start()

    # get number of IFs
    Nch = indata.header["naxis"][2]
    Nif = indata.header["naxis"][3]

    # get old tables
    oldtabs = indata.tables

    # Run ACSCL
    task = tget("acscl")
    task.getn(indata)
    task.solint=solint
    task.sources=calsour
    task.docalib=1
    task.gainuse=int(gainver)
    task.doband=int(doband)
    task.bpver=int(bpver)
    task.ichansel[1] = [None, 1, Nch, 1, 0]
    task()
    SNver = indata.table_highver("SN")

    if flagmode==1:
        # Flagging data
        task = tget("snedt")
        task.getn(indata)
        task.inext="SN"
        task.invers=SNver
        task()
        SNver = indata.table_highver("SN")
    elif flagmode==2:
        # Flagging data
        task = tget("edita")
        task.getn(indata)
        task.inext="SN"
        task.invers=SNver
        task.outfgver=indata.table_highver("FG")+1
        task.reason="BAD ACSCL SOLUTION"
        task()
        ehtfgreset(indata, fgver=0)

    # CLCAL
    task = tget("clcal")
    task.getn(indata)
    task.sources=sources
    task.calsour=calsour
    task.interpol="2PT"
    task.snver = SNver
    task.gainver = int(gainver)
    task.gainuse = int(gainuse)
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtacscl: new tables created")
    print(diftables)
    return diftables

# ------------------------------------------------------------------------------
# A-Priori Calibration
# ------------------------------------------------------------------------------
def ehtantab(indata, antabfileNA, antabfileAA1, antabfileAA2):
    '''
    Args:
        indata (AIPSUVData object):
            input data
        antabfileNA, antabfileAA1, antabfileAA2 (str):
            filenames of antab files for Non ALMA station, and ALMA
            (two files splitted by Michael Janssen's split_antab.py).
    '''
    # get old tables
    oldtabs = indata.tables

    # run antab
    task.tyver=indata.table_highver("TY")
    task.gcver=indata.table_highver("GC")
    task.calin=antabfileAA1
    task.check()
    task()
    task.tyver=indata.table_highver("TY")
    task.gcver=indata.table_highver("GC")
    task.calin=antabfileAA2
    task.check()
    task()
    task = tget("antab")
    task.getn(indata)
    task.tyver=indata.table_highver("TY")+1
    task.gcver=indata.table_highver("GC")+1
    task.calin=antabfileNA
    task.check()
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtantab: new tables created")
    print(diftables)
    return diftables

def ehtapcal(indata, tyver=0, gcver=0, dif=8):
    '''
    '''
    # get old tables
    oldtabs = indata.tables

    # run APCAL
    Nif = indata.header.naxis[3]
    SNver0 = indata.table_highver("SN")+1
    for bif in range(1,Nif+1,dif):
        task=tget("apcal")
        task.getn(indata)
        task.bif=bif
        task.eif=int(np.min([bif+dif,Nif]))
        task.tyver=tyver
        task.gcver=gcver
        task.dofit[1]=-1
        task()
    SNver1 = indata.table_highver("SN")+1

    # run CLCAL
    task=tget("clcal")
    task.getn(indata)
    task.interpol="SELF"
    task.snver=SNver0
    task.invers=SNver1
    task.gainver=indata.table_highver("CL")
    task.gainuse=indata.table_highver("CL")+1
    task.refant=-1
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtapcal: new tables created")
    print(diftables)
    return diftables

# ------------------------------------------------------------------------------
# Band Pass
# ------------------------------------------------------------------------------
def ehtbpassac(
        indata,
        calsour=[None, ""],
        timerang=[None,0,0,0,0,0,0,0,0],
        docal=-1,
        gainuse=-1,
        doband=-1,
        bpver=-1,
        solint=-1):
    '''
    '''

    # get old tables
    oldtabs = indata.tables

    # Run ACSCL
    task = tget("bpass")
    task.getn(indata)
    task.calsour=calsour
    task.timerang=timerang
    task.docalib=int(docal)
    task.gainuse=int(gainuse)
    task.doband=int(doband)
    task.bpver=int(bpver)
    task.bpassprm[1]=1
    task.bpassprm[10]=1
    task.solint=solint
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtbpassac: new tables created")
    print(diftables)
    return diftables


def ehtbpasscc(
        indata,
        calsour=[None, ""],
        timerang=[None,0,0,0,0,0,0,0,0],
        docal=-1,
        gainuse=-1,
        doband=-1,
        bpver=-1,
        solint=-1):
    '''
    '''

    # get old tables
    oldtabs = indata.tables

    # Run ACSCL
    task = tget("bpass")
    task.getn(indata)
    task.calsour=calsour
    task.timerang=timerang
    task.docalib=int(docal)
    task.gainuse=int(gainuse)
    task.doband=int(doband)
    task.bpver=int(bpver)
    task.bpassprm[1]=0
    task.bpassprm[10]=1
    task.solint=solint
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtbpasscc: new tables created")
    print(diftables)
    return diftables


# ------------------------------------------------------------------------------
# SNPLT
# ------------------------------------------------------------------------------
def ehtsnplt(
        indata,
        inext="SN",
        invers=0,
        pltifs = 1,
        tmode = "",
        stokes="",
        timerang=[None,0,0,0,0,0,0,0,0],
        antennas=[None,0],
        eachant=False,
        optypes = "",
        opcode = "ALSI",
        do3col=1,
        nplots=1,
        outfile = "",
        pagetype=0,
        overwrite = True,
        ignoreruntimeerr=True,
        zappltabs=False):
    '''
    '''

    Nant = len(indata.antennas)
    oldtabs = indata.tables

    # TIMERANG
    if tmode.lower() == "eachscan":
        scantimes = indata.scantimes()
    else:
        scantimes = [timerang]

    # BIF, EIF
    if np.isscalar(pltifs):
        pltifs = np.array([pltifs])
    bif = pltifs.min()
    eif = pltifs.max()

    # BIF, EIF
    if np.isscalar(optypes):
        optypes = [optypes]

    if eachant:
        if antennas[1]==0:
            antlist=np.arange(Nant)+1
        else:
            antlist=antennas[1:]
    else:
        antlist=[1]

    # PLver
    PLverbef = indata.table_highver("PL")
    for optype in optypes:
        for antenna in antlist:
            for scantime in scantimes:
                task = tget("snplt")
                task.getn(indata)
                task.inext=inext.upper()
                task.invers=invers
                if eachant:
                    task.antennas[1]=int(antenna)
                else:
                    task.antennas=antennas
                task.timerang=scantime
                task.stokes=stokes
                task.bif=int(bif)
                task.eif=int(eif)
                task.optype=optype.upper()
                task.opcode=opcode.upper()
                if eachant:
                    task.nplots=1
                else:
                    task.nplots=nplots
                task.dotv=-1
                task.do3col=do3col
                try:
                    task()
                except RuntimeError:
                    pass
                except OSError:
                    pass
    PLveraft = indata.table_highver("PL")

    task = tget("lwpla")
    task.getn(indata)
    task.plver=int(PLverbef+1)
    task.invers=int(PLveraft)
    task.dparm[5]=2
    task.dparm[6]=int(pagetype)
    task.outfile=outfile
    task.set_plcolors_aipstv()
    task.check(overwrite=overwrite)
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtsnplt: new tables created")
    print(diftables)
    if zappltabs:
        print("ehtsnplt: removing PL tables")
        indata.extdest(diftables)


# ------------------------------------------------------------------------------
# POSSM
# ------------------------------------------------------------------------------
def ehtpossmac(
        indata,
        docalib=-1,
        gainuse=0,
        flagver=0,
        doband=-1,
        bpver=-1,
        solint=-1,
        bindif=True,
        bindpol=True,
        plotbp=False,
        outfile = "",
        pagetype=0,
        overwrite=True,
        zappltabs=False):
    '''
    '''

    # get some info
    Nant = len(indata.antennas)
    Nif = indata.header["naxis"][3]
    oldtabs = indata.tables

    # set parameters
    if bindif:
        bifs=[1]
        eifs=[0]
    else:
        bifs=np.arange(Nif)+1
        eifs=bifs
    Npltif = len(bifs)

    if bindpol:
        stokess=["RRLL"]
    else:
        stokess=["RR","LL"]

    # PLver
    PLverbef = indata.table_highver("PL")
    for iant in range(Nant):
        for stokes in stokess:
            for iif in range(Npltif):
                task = tget("possm")
                task.getn(indata)
                task.stokes=stokes
                task.antennas[1]=iant+1
                task.baseline[1]=iant+1
                task.bif=int(bifs[iif])
                task.eif=int(eifs[iif])
                task.docalib=int(docalib)
                task.gainuse=int(gainuse)
                task.flagver=int(flagver)
                task.doband=int(doband)
                task.bpver=int(bpver)
                # Bandpass or AC
                if plotbp:
                    task.aparm[8]=2
                else:
                    task.aparm[8]=1
                # Bind IF / Pol
                if bindif and bindpol:
                    task.aparm[9]=3
                elif bindif:
                    task.aparm[9]=1
                elif bindpol:
                    task.aparm[9]=2
                task.codetype="AMP"
                task.solint=solint
                task.nplots=1
                task.dotv=-1
                task()
    PLveraft = indata.table_highver("PL")

    task = tget("lwpla")
    task.getn(indata)
    task.plver=int(PLverbef+1)
    task.invers=int(PLveraft)
    task.dparm[5]=2
    task.dparm[6]=int(pagetype)
    task.outfile=outfile
    task.set_plcolors_aipstv()
    task.check(overwrite=overwrite)
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtpltacspr: new tables created")
    print(diftables)
    if zappltabs:
        print("ehtpltacspr: removing PL tables")
        indata.extdest(diftables)


def ehtpossmcc(
        indata,
        docalib=-1,
        gainuse=0,
        flagver=0,
        doband=-1,
        bpver=-1,
        solint=-1,
        bindif=True,
        plotbp=False,
        outfile = "",
        pagetype=0,
        overwrite=True,
        zappltabs=False):
    '''
    '''

    # get some info
    Nant = len(indata.antennas)
    Nif = indata.header["naxis"][3]
    oldtabs = indata.tables

    # set parameters
    if bindif:
        bifs=[1]
        eifs=[0]
    else:
        bifs=np.arange(Nif)+1
        eifs=bifs
    Npltif = len(bifs)

    # PLver
    PLverbef = indata.table_highver("PL")
    for iant in range(Nant):
        if iant == 0:
            continue
        for iif in range(Npltif):
            task = tget("possm")
            task.getn(indata)
            task.antennas[1]=1
            task.baseline[1]=iant+1
            task.bif=int(bifs[iif])
            task.eif=int(eifs[iif])
            task.docalib=int(docalib)
            task.gainuse=int(gainuse)
            task.flagver=int(flagver)
            task.doband=int(doband)
            task.bpver=int(bpver)
            if plotbp:
                task.aparm[8]=2
            else:
                task.aparm[8]=0
            if bindif:
                task.aparm[9]=3
            task.solint=solint
            task.nplots=1
            task.dotv=-1
            task()
    PLveraft = indata.table_highver("PL")

    task = tget("lwpla")
    task.getn(indata)
    task.plver=int(PLverbef+1)
    task.invers=int(PLveraft)
    task.dparm[5]=2
    task.dparm[6]=int(pagetype)
    task.outfile=outfile
    task.set_plcolors_aipstv()
    task.check(overwrite=overwrite)
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtpltacspr: new tables created")
    print(diftables)
    if zappltabs:
        print("ehtpltacspr: removing PL tables")
        indata.extdest(diftables)


# ------------------------------------------------------------------------------
# VPLOT
# ------------------------------------------------------------------------------
def ehtvplotac(
        indata,
        pltifs = 1,
        tmode = "",
        timerang=[None,0,0,0,0,0,0,0,0],
        docalib=-1,
        gainuse=0,
        flagver=0,
        doband=-1,
        bpver=-1,
        opcode = "",
        outfile = "",
        pagetype=0,
        overwrite = True,
        zappltabs=False):
    '''
    Plot CH-averaged autocorrelation power.
    This function runs vplot for multple times to plot autocorrelation power
    on specified IF(s).

    Args:
      indata (AIPSUVData object):
        input data
      pltifs (integer, or array of integers; default=1):
        IF(s) to be plotted
      tmode (str, default=""):
        if tmode=="eachscan", vplot is run for each scan defined in the NX table.
        Othewise, it will plot data on timerange specified by "timerang"
      timerang (arraylike, default=[None,0,0,0,0,0,0,0,0]):
        Timeranges when data will be plotted. if tmode=="eachscan", this parameter
        will be ignored
      docalib (integer, default=-1):
        if docalib=1, calibration tables will be applied. see docalib in vplot;
    '''
    # get number of IFs
    Nch = indata.header["naxis"][2]
    Nif = indata.header["naxis"][3]
    Nant = len(indata.antennas)
    oldtabs = indata.tables

    if tmode.lower() == "eachscan":
        scantimes = indata.scantimes()
    else:
        scantimes = [timerang]

    if np.isscalar(pltifs):
        pltifs = np.array([pltifs])

    # PLver
    PLverbef = indata.table_highver("PL")
    for iant in range(Nant):
        for scantime in scantimes:
            for pltif in pltifs:
                bif = pltif
                eif = pltif
                task = tget("vplot")
                task.getn(indata)
                task.bchan=1
                task.echan=Nch
                task.avgchan=1
                task.crowded=1
                task.do3col=1
                task.bif=int(bif)
                task.eif=int(eif)
                task.timerang=scantime
                task.stokes="RRLL"
                task.antennas[1]=iant+1
                task.baseline[1]=iant+1
                task.docalib=int(docalib)
                task.gainuse=int(gainuse)
                task.flagver=int(flagver)
                task.doband=int(doband)
                task.bpver=int(bpver)
                task.optype="AUTO"
                task.opcode=opcode.upper()
                task.nplots=1
                task.dotv=-1
                task()
    PLveraft = indata.table_highver("PL")

    task = tget("lwpla")
    task.getn(indata)
    task.plver=int(PLverbef+1)
    task.invers=int(PLveraft)
    task.dparm[5]=2
    task.dparm[6]=int(pagetype)
    task.outfile=outfile
    task.set_plcolors_aipstv()
    task.check(overwrite=overwrite)
    task()

    newtabs = indata.tables
    diftables = difftables(oldtabs, newtabs)
    print("ehtpltacp: new tables created")
    print(diftables)
    if zappltabs:
        print("ehtpltacspr: removing PL tables")
        indata.extdest(diftables)
