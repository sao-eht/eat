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

