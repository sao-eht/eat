#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A python module eat.aips.aipstask

This is a submodule of eat.aips. This submodule contains EHT-specific functions
for AIPS reduction using ParselTongue.

This module must be loaded after rnnning eat.aips.set_env().
'''

# Check if ParselTongue modeules can be load.
import eat.aips as ea
ea.check_pt(printver=False)
#
import pandas as pd
import numpy as np
#
from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList
from AIPSData import AIPSUVData, AIPSImage, AIPSCat
from AIPSTV import AIPSTV
import eat.aips.aipstask as aipstask

def ehtload(outdata,
            datain="",
            ncount=1000,
            clint=1/60.):
    aipstask.zap(outdata)
    task = aipstask.AIPSTask("fitld")
    task.defaults()
    task.geton(outdata)
    task.datain=datain
    task.ncount=ncount
    task.doconcat=1
    task.clint=clint
    task.check()
    task()


def ehtsort(indata, outdata,
            clint=1/60.):
    aipstask.zap(outdata)
    task = aipstask.AIPSTask("msort")
    task.defaults()
    task.getn(indata)
    task.geton(outdata)
    task()

    task = aipstask.AIPSTask("indxr")
    task.defaults()
    task.getn(outdata)
    task.cparm[3] = clint
    task()


def ehtsumm(indata,
            docrt=-1, 
            prtanout="prtan.txt",
            listrout="listr.txt",
            dtsumout="dtsum.txt",
            overwrite=False):
    # PRTAN
    task = aipstask.AIPSTask("prtan")
    task.defaults()
    task.getn(indata)
    task.docrt=docrt
    if task.docrt < 0:
        task.outprint=prtanout
        task.check(overwrite=overwrite)
    task()


    # LISTR (SCAN)
    task = aipstask.AIPSTask("listr")
    task.defaults()
    task.getn(indata)
    task.optype='SCAN'
    task.docrt=docrt
    if task.docrt < 0:
        task.outprint=listrout
        task.check(overwrite=overwrite)
    task()


    # DTSUM
    task = aipstask.AIPSTask("dtsum")
    task.defaults()
    task.getn(indata)
    task.aparm[1]=2
    task.docrt=docrt
    if task.docrt < 0:
        task.outprint=dtsumout
        task.check(overwrite=overwrite)
    task()


def ehtancor(indata,
             inver=0,
             datain=""):
    # Get correction Table
    cortable = pd.read_csv(datain)
    print(cortable)
    annames_tab = list(set(cortable["ANNAME"]))
    
    # Get Antennna information
    annames = indata.antennas
    Nan = len(annames)
    for ian in range(Nan):
        anname = annames[ian]
        if anname not in annames_tab:
            print("[WARNING] No correction info for the station %s"%(anname))
            continue
        mntsta = cortable.loc[cortable["ANNAME"]==anname,"MNTSTA"].reset_index(drop=True)[0]
        #
        task = aipstask.AIPSTask("tabed")
        task.defaults()
        task.getn(indata)
        task.inext = 'AN'
        task.optype = 'REPL'
        task.inver = inver
        task.outver = -1
        task.aparm[1] = 5
        task.aparm[4] = 4
        task.bcount = int(ian+1)
        task.ecount = int(ian+1)
        task.keyvalue[1] = int(mntsta)
        #print(ian+1,anname,mntsta)
        task()
