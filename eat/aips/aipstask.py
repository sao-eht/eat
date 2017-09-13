#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A python module eat.aips.aipstask

This is a submodule of eat.aips. This submodule contains useful functions to
run AIPStasks, handle AIPS data using ParselTongue.

This module must be loaded after rnnning eat.aips.set_env().
'''


# Check if ParselTongue modeules can be load.
import eat.aips as ea
ea.check_pt(printver=False)

# Load modules
import os
import numpy as np
import datetime

from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList
from AIPSData import AIPSUVData, AIPSImage, AIPSCat
from AIPSTV import AIPSTV

# Overwrite AIPSTask
class AIPSTask(AIPSTask, object):
    userno = -1
    def getn(self,uvdata):
        dirs = dir(self)
        if 'inname' in dirs:
            self.inname = uvdata.name
            self.inclass = uvdata.klass
            self.inseq = uvdata.seq
            self.indisk = uvdata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of getn')

    def get2n(self,uvdata):
        dirs = dir(self)
        if 'in2name' in dirs:
            self.in2name = uvdata.name
            self.in2class = uvdata.klass
            self.in2seq = uvdata.seq
            self.in2disk = uvdata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of get2n')

    def get3n(self,uvdata):
        dirs = dir(self)
        if 'in3name' in dirs:
            self.in3name = uvdata.name
            self.in3class = uvdata.klass
            self.in3seq = uvdata.seq
            self.in3disk = uvdata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of get3n')

    def get4n(self,uvdata):
        dirs = dir(self)
        if 'in4name' in dirs:
            self.in4name = uvdata.name
            self.in4class = uvdata.klass
            self.in4seq = uvdata.seq
            self.in4disk = uvdata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of get4n')

    def geton(self,uvdata):
        dirs = dir(self)
        if 'outname' in dirs:
            self.outname = uvdata.name
            self.outclass = uvdata.klass
            self.outseq = uvdata.seq
            self.outdisk = uvdata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of geton')

    def set_params(self, **args):
        keys = args.keys()
        for key in keys:
            try:
                getattr(self, key)
            except AttributeError:
                errmsg="This AIPSTask has no attribute '%s'"%(key)
                raise AttributeError(errmsg)
            setattr(self, key, args[key])

    def check(self, overwrite=True):
        dirs = dir(self)
        if 'datain' in dirs:
            isdatain = os.path.isfile(self.datain)
            isdatain1= os.path.isfile(self.datain+"1")
            if (isdatain is False) and (isdatain1 is False):
                raise ValueError("datain=%s cannot be found."%(self.datain))
        if 'outprint' in dirs:
            if os.path.isfile(self.outprint) and overwrite:
                os.system("rm -f %s"%(self.outprint))
                print("%s was deleted."%(self.outprint))
        if 'outfile' in dirs:
            if os.path.isfile(self.outfile) and overwrite:
                os.system("rm -f %s"%(self.outfile))
                print("%s was deleted."%(self.outfile))


# Override help() such that it prints something useful for instances
# of AIPSTask.
_help = help
def help(obj):
    if isinstance(obj, AIPSTask):
        obj.help()
    elif type(obj)==str:
        AIPSTask(obj).help()
    else:
        _help(obj)
        pass
    return


def explain(obj):
    if isinstance(obj, AIPSTask):
        obj.explain()
    elif type(obj)==str:
        AIPSTask(obj).explain()
    return


def uvdataname(uvdata):
    # shortcut to return the aips catalogue name of an AIPSUVData object
    outstr = uvdata.name
    outstr+= '.' + uvdata.klass
    outstr+= '.' + str(uvdata.seq)
    outstr+= '.' + str(uvdata.disk)
    return outstr


def zap(uvdata):
    # delete AIPS data from catalogue
    if uvdata.exists():
        print("zap %s."%(uvdataname(uvdata)))
        uvdata.zap(force=True)
    else:
        print("%s does not exist."%(uvdataname(uvdata)))


def scantimes(indata):
    '''
    
    '''
    # Check if NX table exists
    NXver = indata.table_highver("AIPS NX")
    if NXver < 1:
        raise ValueError("Input Data have NO index table.")
    NXtab = indata.table("AIPS NX", NXver)
    Nscan = len(NXtab)

    # calc timerang
    keys = NXtab[0].keys()
    scantimes = []
    for iscan in range(Nscan):
        t= NXtab[iscan]["time"]  # Central time of each scan (day)
        dt = NXtab[iscan]["time_interval"]  # Scan length (day)
        
        t1 = t - dt/2.
        t2 = t + dt/2.
        
        d1 = int(np.floor(t1))
        t1 = 24.0 * (t1 - d1)
        h1 = int(np.floor(t1))
        t1 = 60.0 * (t1 - h1)
        m1 = int(np.floor(t1))
        s1 = int(np.ceil(60 * (t1 - m1)))
        
        if s1 >= 60:
            s1 -= 60
            m1 += 1
        if m1 >= 60:
            m1 -= 60
            h1 += 1
        if h1 >= 24:
            h1 -= 24
            d1 += 1
        
        d2 = int(np.floor(t2))
        t2 = 24.0 * (t2 - d2)
        h2 = int(np.floor(t2))
        t2 = 60.0 * (t2 - h2)
        m2 = int(np.floor(t2))
        s2 = int(np.ceil(60 * (t2 - m2)))
        
        if s2 >= 60:
            s2 -= 60
            m2 += 1
        if m2 >= 60:
            m2 -= 60
            h2 += 1
        if h2 >= 24:
            h2 -= 24
            d2 += 1
        
        # Append timerang of each scan
        scantimes.append([None, d1,h1,m1,s1,d2,h2,m2,s2])
    return scantimes


def set_plcolors_aipstv(task):
    if "plcolors" in dir(task):
        task.plcolors[1] = [None, 0., 0., 0.]
        task.plcolors[2] = [None, 0.06275, 1., 0.]
        task.plcolors[3] = [None, 1., 0.6706, 1.]
        task.plcolors[4] = [None, 0., 1., 1.]
        task.plcolors[5] = [None, 1., 1., 1.]
        task.plcolors[6] = [None, 1., 1., 1.]
        task.plcolors[7] = [None, 1., 1., 1.]
        task.plcolors[8] = [None, 1., 1., 1.]
        task.plcolors[9] = [None, 0., 0., 0.]
        task.plcolors[10] = [None, 1., 1., 1.]
        task.docolor=1
    else:
        raise ValueError("The input task does not have an attribute 'plcolors'")
