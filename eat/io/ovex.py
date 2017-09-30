"""
USAGE NOTE:

ovex_reader.Ovex('ovex file name') returns an object that contains
various observational information.

import ovex_reader as ov
fname = "ovex_files/3C279.zmubap"
ovex = ov.Ovex(fname)


Summary of mnemonics:

--- Read in from "$MODE" section in a ovex file
"mode index"=0 if there's only 1 mode
ovex.modes[mode index]['mode'] = name of mode
ovex.modes[mode index]['site_dic'] = dictionary of site name in $FREQ language and site ID in $SITES language

--- Read in from "$FREQ" section in a ovex file
ovex.freqs[site index]['antenna'] = antenna ID
ovex.freqs[site index]['sample_rate'] = sample rate
ovex.freqs[site index]['chans'][channel index]['chan_name'] = name of channel ?
ovex.freqs[site index]['chans'][channel index]['freq'] = frequency in MHz
ovex.freqs[site index]['chans'][channel index]['bw'] = bandwidth in MHz


--- Read in from "$SITES" section in a ovex file
ovex.sites[site index]['site_name']
ovex.sites[site index]['site_ID']
ovex.sites[site index]['mk4_site_ID']
ovex.sites[site index]['site_position']

--- Read in from "$SCHED" section in a ovex file
"scan index"=0 if there's only 1 scan, which is always the case for ovex files ??
ovex.sched[scan index]['scan_numer'] = scan number of "scan ID"-th scan
ovex.sched[scan index]['source'] = source name of "scan ID"-th scan
ovex.sched[scan index]['start'] = start time of the scan
ovex.sched[scan index]['mode'] = scan mode of the scan
ovex.sched[scan index]['scan'][site index]['site'] = site name of "site-ID"-th site of the "scan ID"-th scan
ovex.sched[scan index]['scan'][site index]['scan_sec'] = duration of the scan in second
ovex.sched[scan index]['scan'][site index]['data_size'] = data size (?) of the scan in GB
ovex.sched[scan index]['scan'][site index]['scan_sec_start'] = probably 'start_hr' + 'scan_sec_start'(in sec) is the actual time that the scan starts, but not sure

--- Read in from "$SOURCE" section in a ovex file
"source index"=0 if there's only 1 source
ovex.source[source index]['source'] = source name of "source ID"-th source
ovex.source[source index]['ra'] = RA of the source
ovex.source[source index]['dec'] = DEC of the source
ovex.source[source index]['ref_coord_frame'] = something about the source

--- Others
ovex.sites_dic = sites' names in various languages (site_name, site_ID, mk4_site_ID, $FREQ's ref)
ovex.lvex_rev
ovex.evex_rev
ovex.ivex_rev
"""
from __future__ import print_function


# from builtins import range
# from builtins import object
import numpy as np
import re
#import jdcal


class Ovex(object):

    def __init__(self, filename):

        f = open(filename)
        raw = f.readlines()
        f.close()

        self.filename = filename

        # Divide 'raw' data into sectors of '$' marks
        # ASSUMING '$' is the very first character in a line (no space in front)
        metalist = [] # meaning list of metadata

        for i in range(len(raw)):
            if raw[i][0]=='$':
                temp = [raw[i]]
                break

        for j in range(i+1,len(raw)):
            if raw[j][0]!='$':
                temp.append(raw[j])
            elif raw[j][0]=='$':
                metalist.append(temp)
                temp = [raw[j]]
            else:
                print('Something is wrong.')
        metalist.append(temp) # don't forget to add the final one
        self.metalist = metalist

        # MODE ==========================================================
        MODE = self.get_sector('MODE')
        modes = []

        indef = False
        for i in range(len(MODE)):

            line = MODE[i]

            if indef==False:
                idx = self.find_variable('def',line)
                if idx>=0:
                    temp = {}
                    temp['mode'] = self.get_def_name(line)
                    temp['site_dic'] = {}
                    cnt = 0
                    indef=True

            if indef==True:
                idx = self.find_variable('ref',line)
                var = self.get_ref_special('$FREQ',line)
                if var!=False:
                    temp['site_dic'][cnt] = var
                    cnt += 1

            if indef==True:
                idx = self.find_variable('enddef',line)
                if idx>=0:
                    modes.append(temp)
                    indef=False
        self.modes = modes


        # Extract desired information
        # SOURCE ========================================================
        SOURCE = self.get_sector('SOURCE')
        source = []
        indef = False

        for i in range(len(SOURCE)):

            line = SOURCE[i]
            if line[0:3]=="def":
                indef=True

            if indef:
                ret = self.get_variable("source_name",line)
                if len(ret)>0: source_name = ret
                ret = self.get_variable("ra",line)
                if len(ret)>0: ra = ret
                ret = self.get_variable("dec",line)
                if len(ret)>0: dec = ret
                ret = self.get_variable("ref_coord_frame",line)
                if len(ret)>0: ref_coord_frame = ret

                if line[0:6]=="enddef":
                    source.append({'source':source_name,'ra':ra,'dec':dec,'ref_coord_frame':ref_coord_frame})
                    indef=False

        self.source = source


        # FREQ ==========================================================
        FREQ = self.get_sector('FREQ')
        freq = []
        indef = False

        for i in range(len(FREQ)):

            line = FREQ[i]

            if indef==False:
                idx = self.find_variable('def',line)
                if idx>=0:
                    temp = {}
                    temp['chans'] = {}
                    cnt = 0
                    temp['antenna'] = self.get_def_name(line)
                    indef=True

            if indef:
                idx = line.find('chan_def')
                if idx>=0:
                    chan_name = self.get_variable("chan_def",line)
                    chan_def = re.findall("[-+]?\d+[\.]?\d*",line)
                    temp['chans'][cnt] = {'chan_name':chan_name,'freq':float(chan_def[1]),'bw':float(chan_def[2])} #MHz
                    cnt+=1
                ret = self.get_variable("sample_rate",line)
                if len(ret)>0:
                    temp['sample_rate'] = ret

            if indef==True:
                idx = self.find_variable('enddef',line)
                if idx>=0:
                    freq.append(temp)
                    indef=False

        self.freqs = freq


        # SITE ==========================================================
        SITE = self.get_sector('SITE')
        sites = []
        site_ID_dict = {}
        indef = False

        for i in range(len(SITE)):

            line = SITE[i]
            if line[0:3]=="def": indef=True

            if indef:
                # get site_name
                ret = self.get_variable("site_name",line)
                if len(ret)>0:
                    site_name = ret

                # get site_ID and make dictionrary for site_ID <-> site_name
                ret = self.get_variable("site_ID",line)
                if len(ret)>0:
                    site_ID = ret
                    site_ID_dict[ret] = site_name

                # get site_position
                ret = self.get_variable("site_position",line)
                if len(ret)>0:
                    site_position = re.findall("[-+]?\d+[\.]?\d*",line)

                # get mk4_site_ID
                ret = self.get_variable("mk4_site_ID",line)
                if len(ret)>0:
                    mk4_site_ID = ret

                # append to "sites"
                if line[0:6]=="enddef":
                    #sites.append([site_name,site_ID,mk4_site_ID,site_position[0],site_position[1],site_position[2]])
                    sites.append({'site_name':site_name,'site_ID':site_ID,'mk4_site_ID':mk4_site_ID,'site_position':site_position})
                    indef=False

        self.sites = sites


        # SCHED  =========================================================
        SCHED = self.get_sector('SCHED')
        sched = []
        inscan = False

        for i in range(len(SCHED)):

            line = SCHED[i]
            if line[0:4]=="scan":
                inscan=True
                temp={}
                temp['scan_number'] = re.findall("[-+]?\d+[\.]?\d*",line)
                temp['scan']={}
                cnt = 0

            if inscan:
                ret = self.get_variable("start",line)
                if len(ret)>0:
                    #mjd,hr = self.vexdate_to_MJD_hr(ret) # convert vex time format to mjd and hour
                    #temp['mjd_floor'] = mjd
                    #temp['start_hr'] = hr
                    temp['start'] = ret

                ret = self.get_variable("mode",line)
                if len(ret)>0: temp['mode'] = ret

                ret = self.get_variable("source",line)
                if len(ret)>0: temp['source'] = ret

                ret = self.get_variable("station",line)
                if len(ret)>0:
                    site_ID = ret
                    site_name = site_ID_dict[site_ID] # convert to more familier site name
                    sdur = re.findall("[-+]?\d+[\.]?\d*",line)
                    s_st = float(sdur[0]) # start time in sec
                    s_en = float(sdur[1]) # end time in sec
                    d_size = float(sdur[2]) # data size(?) in GB
                    temp['scan'][cnt] = {'site':site_name,'scan_sec_start':s_st,'scan_sec':s_en,'data_size':d_size}
                    cnt +=1

                if line[0:7]=="endscan":
                    sched.append(temp)
                    inscan=False

        self.sched = sched


        # LVEX_REV =====================================================
        LVEX_REV = self.get_sector('LVEX_REV')

        for i in range(len(LVEX_REV)):
            line = LVEX_REV[i]
            ret = self.get_variable("rev",line)
            if len(ret)>0:
                self.lvex_rev = float(ret)


        # EVEX_REV =====================================================
        EVEX_REV = self.get_sector('EVEX_REV')

        for i in range(len(EVEX_REV)):
            line = EVEX_REV[i]
            ret = self.get_variable("rev",line)
            if len(ret)>0:
                self.evex_rev = float(ret)


        # IVEX_REV =====================================================
        IVEX_REV = self.get_sector('IVEX_REV')

        for i in range(len(IVEX_REV)):
            line = IVEX_REV[i]
            ret = self.get_variable("rev",line)
            if len(ret)>0:
                self.ivex_rev = float(ret)


        # list of various ways of calling the same site ================
        sites_dic = []
        for i in range(len(self.modes[0]['site_dic'])):
            id0 = site_ID_dict[self.modes[0]['site_dic'][i][1]]
            id1 = self.modes[0]['site_dic'][i][1]
            for j in range(len(self.sites)):
                if self.sites[j]['site_ID'] == id1:
                    id2 = self.sites[j]['mk4_site_ID']
            id3 = self.modes[0]['site_dic'][i][0]
            sites_dic.append([id0,id1,id2,id3])
        self.sites_dic = sites_dic


    # ====================================================================
    # ====================================================================


    # Function to obtain a desired sector from 'metalist'
    def get_sector(self, sname):
        for i in range(len(self.metalist)):
            if sname in self.metalist[i][0]:
                return self.metalist[i]
        print('No sector named %s'%sname)
        return False

    # Function to get a value of 'vname' in a line which has format of
    # 'vname' = value ;(or :)
    def get_variable(self, vname, line):
        idx = self.find_variable(vname,line)
        name = ''
        if idx>=0:
            start = False
            for i in range(idx+len(vname),len(line)):
                if start==True:
                    if line[i]==';' or line[i]==':': break
                    elif line[i]!=' ': name += line[i]
                if start==False and line[i]!=' ' and line[i]!='=': break
                if line[i]=='=': start = True
        return name

    # Function to find MJD (int!) and hour in UT from vex format,
    # e.g, 2016y099d05h00m00s
    """
    def vexdate_to_MJD_hr(self, vexdate):
        time = re.findall("[-+]?\d+[\.]?\d*",vexdate)
        year = int(time[0])
        date = int(time[1])
        mjd = jdcal.gcal2jd(year,1,1)[1]+date-1
        hour = int(time[2]) + float(time[3])/60. + float(time[4])/60./60.
        return mjd,hour
    """

    # check if a variable 'vname' exists by itself in a line.
    # returns index of vname[0] in a line, or -1
    def find_variable(self, vname, line):
        idx = line.find(vname)
        if ((idx>0 and line[idx-1]==' ') or idx==0) and line[0]!='*':
            if idx+len(vname)==len(line): return idx
            if line[idx+len(vname)]=='=' or line[idx+len(vname)]==' ' or line[idx+len(vname)]==':' or line[idx+len(vname)]==';': return idx
        return -1

    # extract def 'def_name'; (or :)
    def get_def_name(self, line):
        idx = self.find_variable('def', line)
        if idx<0: return False

        var = ''
        invar = False
        for i in range(idx+3,len(line)):
            if invar == True and (line[i]==' ' or line[i]==';' or line[i]==':'): return var
            if line[i]!=' ': invar = True
            if invar==True: var += line[i]
        return False

    # extract 'ref $FREQ = ant00:Aa;' kind of format
    def get_ref_special(self, vname, line):
        idx = self.find_variable('ref', line)
        if idx<0: return False
        line = line[idx+3:]

        idx = self.find_variable(vname, line)
        if idx<0: return False

        var1 = self.get_variable(vname,line)

        idx = line.find(var1)
        for i in range(idx+len(var1),len(line)):
            if line[i]==';' or line[i]==':': break
        idx = i+1

        var2 = ''
        invar = False
        for i in range(idx,len(line)):
            if invar == True and (line[i]==' ' or line[i]==';' or line[i]==':'): break
            if invar == False and line[i]!=' ': invar = True
            if invar==True: var2 += line[i]
        return var1,var2
