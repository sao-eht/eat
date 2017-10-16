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
ea.check(printver=False)

# Load modules
import os
import numpy as np
import datetime

from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList
from AIPSData import AIPSUVData, AIPSImage, AIPSCat
from AIPSTV import AIPSTV

# ------------------------------------------------------------------------------
# Overwrite Some Parsel Tongue Functions
# ------------------------------------------------------------------------------
class AIPSTV(AIPSTV, object):
    def restart(self, force=False):
        if self.exists():
            if force:
                self.kill()
                self.__init__()
                self.start()
            else:
                print("TV is runnning!")
        else:
            self.__init__()
            self.start()

class AIPSUVData(AIPSUVData, object):
    def scantimes(self):
        '''
        Get a list of timerange parameters for the entire observations.

        Returns:
            scantimes (list):
                a list of timerang parameters. Each element is in a format
                '[None, d1, h1, m1, s1, d2, h2, m2, s2]'
        '''
        # Check if NX table exists
        NXver = self.table_highver("AIPS NX")
        if NXver < 1:
            raise ValueError("Input Data have NO index table.")
        NXtab = self.table("AIPS NX", NXver)
        Nscan = len(NXtab)

        # calc timerang
        keys = NXtab[0].keys()
        scantimes = []
        for iscan in range(Nscan):
            t= NXtab[iscan]["time"]  # Central time of each scan (day)
            dt = NXtab[iscan]["time_interval"]  # Scan length (day)

            t1 = t - dt/2.
            t2 = t + dt/2.

            scantime = [None]
            scantime += fday2timerang(t1).tolist()
            scantime += fday2timerang(t2).tolist()

            # Append timerang of each scan
            scantimes.append(scantime)
        return scantimes

    def extdest(self, tables):
        '''
        Remove single or multiple tables in a specified self

        Args:
            tables (list):
                table or a list for tables to be removed.
                it must be in one of following formats.
                    1) ["table name", "table ver"]
                    2) ["table ver", "table name"]
                    3) [["table1 ver", "table1 name"], ....]
        '''
        self.clrstat()
        if type(tables[0]) == type([]):
            for table in tables:
                print("Zap %s #%d"%(table[1], table[0]))
                self.zap_table(table[1], table[0])
        elif type(tables[0]) == type(""):
            print("Zap %s #%d"%(tables[0], tables[1]))
            self.zap_table(tables[0], tables[1])
        elif type(tables[0]) == type(1):
            print("Zap %s #%d"%(tables[1], tables[0]))
            self.zap_table(tables[1], tables[0])

    def uvdataname(self):
        '''
        shortcut to return the aips catalogue name of an AIPSUVData object
        '''
        outstr = self.name
        outstr+= '.' + self.klass
        outstr+= '.' + str(self.seq)
        outstr+= '.' + str(self.disk)
        return outstr

    def antennaids(self, antnames):
        '''
        Args:
            antnames (list):
                list of antenna names.
        Returns:
            antennas (list):
                a list of antenna ID. Each element is in a format
                '[ID1, ID2, ID3, ....,]'
        '''
        antennas = self.antennas

        if np.isscalar(antnames):
            return antennas.index(antnames)+1
        else:
            antnums = []
            for antname in antnames:
                antnums.append(antennas.index(antname)+1)
            return antnums


class AIPSTask(AIPSTask, object):
    userno = -1
    def getn(self,uvdata):
        dirs = dir(self)
        if np.isscalar(uvdata):
            indata = getuvdata(int(uvdata))
        else:
            indata = uvdata
        if 'inname' in dirs:
            self.inname = indata.name
            self.inclass = indata.klass
            self.inseq = indata.seq
            self.indisk = indata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of getn')

    def get2n(self,uvdata):
        dirs = dir(self)
        if np.isscalar(uvdata):
            indata = getuvdata(int(uvdata))
        else:
            indata = uvdata
        if 'in2name' in dirs:
            self.in2name = indata.name
            self.in2class = indata.klass
            self.in2seq = indata.seq
            self.in2disk = indata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of get2n')

    def get3n(self,uvdata):
        dirs = dir(self)
        if np.isscalar(uvdata):
            indata = getuvdata(int(uvdata))
        else:
            indata = uvdata
        if 'in3name' in dirs:
            self.in3name = indata.name
            self.in3class = indata.klass
            self.in3seq = indata.seq
            self.in3disk = indata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of get3n')

    def get4n(self,uvdata):
        dirs = dir(self)
        if np.isscalar(uvdata):
            indata = getuvdata(int(uvdata))
        else:
            indata = uvdata
        if 'in4name' in dirs:
            self.in4name = indata.name
            self.in4class = indata.klass
            self.in4seq = indata.seq
            self.in4disk = indata.disk
        else:
            print('[WARNING] This AIPS task does not have adverbs of get4n')

    def geton(self,uvdata):
        dirs = dir(self)
        if np.isscalar(uvdata):
            indata = getuvdata(int(uvdata))
        else:
            indata = uvdata
        if 'outname' in dirs:
            self.outname = indata.name
            self.outclass = indata.klass
            self.outseq = indata.seq
            self.outdisk = indata.disk
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
        '''
        Check Values of Adverbs

        Args:
            ignore (bool, default=True):

        '''
        dirs = dir(self)
        # Datain
        if "datain" in dirs:
            isfile0 = os.path.isfile(self.datain)
            isfile1 = os.path.isfile(self.datain+"1")
            if (isfile0 is False) and (isfile1 is False):
                errmsg = "datain='%s' cannot be found."%(self.datain)
                raise ValueError(errmsg)

        # Input Files
        attrs = ["intext", "calin"]
        for attr in attrs:
            if attr in dirs:
                filename = self.__getattribute__(attr)
                if filename == "":
                    print("CHECK: %s is blank"%(attr))
                else:
                    if not os.path.isfile(filename):
                        errmsg = "%s='%s' cannot be found."%(attr,filename)
                        raise ValueError(errmsg)

        # Output Files
        attrs = ["outprint", "outfile", "outtext", "dataout"]
        for attr in attrs:
            if attr in dirs:
                filename = self.__getattribute__(attr)
                if filename == "":
                    print("CHECK: %s is blank"%(attr))
                else:
                    if os.path.isfile(filename) and overwrite:
                        command = "rm -f %s"%(filename)
                        print("CHECK: %s"%(command))
                        os.system(command)
                        print("CHECK: %s was deleted."%(filename))

        # TV
        if 'dotv' in dirs:
            if self.dotv>0:
                tv = AIPSTV()
                tv.restart()

    def set_plcolors_aipstv(self):
        '''
        set plcolors for lwpla
        '''
        if "plcolors" in dir(self):
            self.plcolors[1] = [None, 0., 0., 0.]
            self.plcolors[2] = [None, 0.06275, 1., 0.]
            self.plcolors[3] = [None, 1., 0.6706, 1.]
            self.plcolors[4] = [None, 0., 1., 1.]
            self.plcolors[5] = [None, 1., 1., 1.]
            self.plcolors[6] = [None, 1., 1., 1.]
            self.plcolors[7] = [None, 1., 1., 1.]
            self.plcolors[8] = [None, 1., 1., 1.]
            self.plcolors[9] = [None, 0., 0., 0.]
            self.plcolors[10] = [None, 1., 1., 1.]
            self.docolor=1
        else:
            raise ValueError("The task does not have an attribute 'plcolors'")


# ------------------------------------------------------------------------------
# Short Cuts
# ------------------------------------------------------------------------------
def tget(taskname):
    '''
    get AIPSTask() object

    Args:
        taskname (str): the name of AIPS Task to be called
    Returns:
        AIPSTask object
    '''
    task = AIPSTask(taskname)
    task.defaults()
    return task

def setuser(userno):
    '''
    set AIPS USER NO

    Args:
        userno (int): AIPS USER NO
    '''
    print("AIPS User No.: %d"%(userno))
    AIPS.userno=int(userno)

def pcat(output=False, doprint=True, disk=1):
    '''
    set AIPS USER NO

    Args:
        output (bool, default=False):
            if True, return a list of AIPS data in the specified disk
        doprint (bool, default=True):
            print list of AIPS data in the specified disk
        disk (int, default=1):
            Number of AIPS Disk
    Returns:
        catalogue (list, if output=True):
            A list of dictionary for uvdata in specified disk
    '''
    if AIPS.userno==0:
        raise ValueError("Please set AIPS USER NO with setuser(XX) or AIPS.userno=XX")

    try:
        catlist = AIPSCat()[disk]
    except KeyError:
        raise ValueError("disk=%d does not exist."%(disk))


        return catlist

    if doprint:
        if len(catlist) == 0:
            if doprint: print("pcat: disk=%d is empty."%(disk))
        else:
            for catdata in catlist:
                print("%3d: %12s.%6s.%3d.%3d %s %s %s"%(
                        catdata["cno"],
                        catdata["name"],
                        catdata["klass"],
                        catdata["seq"],
                        disk,
                        catdata["type"],
                        catdata["time"],
                        catdata["date"]))

    if output:
        return catlist


def getuvdata(cno,disk=1):
    '''
    get AIPSUVData object at a specified AIPS Catalogue number and AIPS Disk.

    Args:
        cno (int):
            AIPS Catalogue Number
        disk (int, default=1):
            Number of AIPS Disk
    Returns:
        AIPSUVData object
    '''
    cat = pcat(output=True, doprint=False, disk=disk)
    isdata = False
    for catdata in cat:
        if catdata["cno"] == cno:
            isdata=True
            break
    if not isdata:
        raise ValueError("No data at cno=%d and disk=%d"%(cno, disk))
    return AIPSUVData(catdata["name"], catdata["klass"], disk, catdata["seq"])


def zap(uvdata):
    '''
    zap uvdata

    Args:
        uvdata (AIPSUVData):
            input uvdata to be zapped
    '''
    # delete AIPS data from catalogue
    if uvdata.exists():
        print("zap %s."%(uvdata.uvdataname()))
        uvdata.zap(force=True)
    else:
        print("%s does not exist."%(uvdata.uvdataname()))


# ------------------------------------------------------------------------------
# Help & Explain
# ------------------------------------------------------------------------------
# Override help() such that it prints something useful for instances
# of AIPSTask.
_help = help
def help(obj):
    if isinstance(obj, AIPSTask):
        obj.help()
    elif type(obj)==type(""):
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


# ------------------------------------------------------------------------------
# Combinient Tools
# ------------------------------------------------------------------------------
def difftables(oldtables, newtables):
    '''
    Take a difference between two tables and output a list of added tables
    in new data sets.

    Args:
        oldtables (list):
            Ouput of uvdata.tables for an old data.
        newtables (list):
            Output of uvdata.tables for a new data.
    Returns:
        difftables (list):
            a list for tables newly added in newtabs.
    '''
    difftab = []
    for newtable in newtables:
        if newtable not in oldtables:
            difftab.append(newtable)
    return difftab


def fday2timerang(t):
    '''
    Convert a fractional day to a timerang parameter.

    Args:
        t (float):
            fractional day
    Returns:
        timerang (list):
            A timerang parameter in a format '[d1, h1, m1, s1]'
    '''
    d1 = int(np.floor(t))
    t1 = 24.0 * (t - d1)
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

    return np.asarray([d1,h1,m1,s1], dtype=np.int64)
