import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# CONCAT TSYS TABLE
# ------------------------------------------------------------------------------
def remove_ifs(tsysdata,bif=2,eif=32):
    import copy
    outdata = copy.deepcopy(tsysdata)

    # columns
    orgcols = list(tsysdata["INDEX"])

    # FOR ALMA
    if  len(orgcols)>2:
        modcols = ["DOY","TIME"]+orgcols[bif-1:eif]
        aftcols = ["DOY","TIME"]+orgcols[0:eif-bif+1]
        outdata["INDEX"] = orgcols[0:eif-bif+1]
        # modify data
        outdata["DATA"] = pd.DataFrame()
        for i in range(len(aftcols)):
            aftcol = aftcols[i]
            modcol = modcols[i]
            outdata["DATA"][aftcol] = tsysdata["DATA"][modcol]
        outdata["DATA"] = outdata["DATA"].reset_index()
    elif len(orgcols)<=10:
        modcols = ["DOY","TIME"]+orgcols
        Nidx = len(orgcols)
        aftcols = ["DOY","TIME"]
        if "32" in orgcols[0]:
            for i in np.arange(Nidx):
                aftcols.append(orgcols[i].replace("32","%d"%(eif-bif+1)))
        elif "64" in orgcols[0]:
            for i in np.arange(Nidx):
                aftcols.append(orgcols[i].replace("64","%d"%(eif-bif+1)))
        outdata["INDEX"] = aftcols[2:]

        # modify data
        outdata["DATA"] = pd.DataFrame()
        for i in range(len(aftcols)):
            aftcol = aftcols[i]
            modcol = modcols[i]
            outdata["DATA"][aftcol] = tsysdata["DATA"][modcol]
        outdata["DATA"] = outdata["DATA"][aftcols]
    return outdata

def apply_tsys(tsysdata, year=2017):
    import copy
    import astropy.time as at
    outdata = copy.deepcopy(tsysdata)

    # Get index
    indexes = outdata["INDEX"]
    cols = ["DOY","TIME"] + indexes

    if tsysdata["FT"] is not None:
        for index in indexes:
            outdata["DATA"].loc[:, index] *= tsysdata["FT"]
        outdata["FT"] = None

    if tsysdata["TIMEOFF"] is not None:
        timetags = get_datetime(outdata["DATA"], year)
        timetags = at.Time(timetags, scale="utc")
        timetags+= at.TimeDelta(tsysdata["TIMEOFF"], format="sec")
        timetags = timetags.yday
        doy = []
        hms = []
        for timetag in timetags:
            dhms = timetag.split(":")
            doy.append(np.int64(dhms[1]))
            hms.append(":".join(dhms[2:]))
        outdata["TIMEOFF"] = None
        outdata["DATA"].loc[:, "DOY"] = np.asarray(doy)
        outdata["DATA"].loc[:, "TIME"] = np.asarray(hms)
    return outdata

def concat_tsys(tsysdata1,tsysdata2,pad_IF_lower=False, year=2017):
    import copy
    tsysdata3 = {}

    # CHECK IF
    for key in ["GROUP", "ANT"]:
        if tsysdata1[key] != tsysdata2[key]:
            raise ValueError("Concat Tsys: %s CODES are different."%(key))
        else:
            tsysdata3[key] = copy.deepcopy(tsysdata1[key])

    # Check TIMEOFF and RANGE
    for key in ["TIMEOFF", "RANGE", "FT"]:
        if tsysdata1[key] != tsysdata2[key]:
            raise ValueError("Concat Tsys: Current function cannot concat tsysdata with different %s."%(key))
        else:
            tsysdata3[key] = copy.deepcopy(tsysdata1[key])

    # Check INDEX
    idx1 = copy.deepcopy(tsysdata1["INDEX"])
    idx2 = copy.deepcopy(tsysdata2["INDEX"])
    Nidx1 = len(tsysdata1["INDEX"])
    Nidx2 = len(tsysdata2["INDEX"])
    data1_pcol = list(tsysdata1["DATA"].columns)
    data2_pcol = list(tsysdata2["DATA"].columns)
    Nrow1 = len(tsysdata1["DATA"]["DOY"])
    Nrow2 = len(tsysdata2["DATA"]["DOY"])
    if Nidx1 == Nidx2:
        tsysdata3["INDEX"] = idx1
    elif Nidx1 > Nidx2:
        tsysdata3["INDEX"] = idx1
        if pad_IF_lower:
            for i in np.arange(Nidx1-Nidx2):
                data2_pcol.insert(2, "pad%d"%(i))
        else:
            for i in np.arange(Nidx1-Nidx2):
                data2_pcol.append("pad%d"%(i))
    else:
        tsysdata3["INDEX"] = idx2
        if pad_IF_lower:
            for i in np.arange(Nidx2-Nidx1):
                data1_pcol.insert(2, "pad%d"%(i))
        else:
            for i in np.arange(Nidx2-Nidx1):
                data1_pcol.append("pad%d"%(i))
    Ndata3_col = np.max([Nidx1,Nidx2])+2
    data3_col = ["DOY", "TIME"] + tsysdata3["INDEX"]
    data1_pad = pd.DataFrame()
    data2_pad = pd.DataFrame()
    for i in np.arange(Ndata3_col):
        if "pad" in data1_pcol[i]:
            data1_pad[data3_col[i]] = np.zeros(Nrow1)
        else:
            data1_pad[data3_col[i]] = tsysdata1["DATA"][data1_pcol[i]]
        if "pad" in data2_pcol[i]:
            data2_pad[data3_col[i]] = np.zeros(Nrow2)
        else:
            data2_pad[data3_col[i]] = tsysdata2["DATA"][data2_pcol[i]]
    data3 = pd.concat([data1_pad, data2_pad], ignore_index=True)
    data3["datetime"] = get_datetime(data3, year)
    data3 = data3.sort_values(by="datetime").reset_index(drop=True)
    tsysdata3["DATA"] = data3[data3_col]
    return tsysdata3


def get_datetime(tsystable, year=2017):
    import astropy.time as at
    Ndata = len(tsystable["DOY"])
    timetags = []
    for idata in np.arange(Ndata):
        doy = tsystable.loc[idata, "DOY"]
        hms = tsystable.loc[idata, "TIME"].split(":")
        tsec = np.float64(hms[0])*3600
        if len(hms)>1:
            tsec += np.float64(hms[1])*60
            if len(hms)>2:
                tsec += np.float64(hms[2])

        timetag = at.Time("%04d:%03d:00:00:00"%(year,doy), scale="utc")
        timetag+= at.TimeDelta(tsec, format="sec")
        timetags.append(timetag.datetime)
    return np.asarray(timetags)

# ------------------------------------------------------------------------------
# READ ANTAB FILE
# ------------------------------------------------------------------------------
def read_antab(filename):
    # Read FILE
    f = open(filename)
    antab = f.readlines()
    f.close()

    # Remove comments
    for iline in np.arange(len(antab)):
        line = antab[iline]
        idx = line.find("!")
        if idx >= 0:
            antab[iline] = line[0:idx]
    antab = "".join(antab)

    # Remove spaces
    antab = antab.upper()
    antab = antab.replace("\n"," ")
    antab = antab.replace("/", " / ")
    while ", " in antab: antab = antab.replace(", ", ",")
    while " ," in antab: antab = antab.replace(" ,", ",")
    while " =" in antab: antab = antab.replace(" =", "=")
    while "= " in antab: antab = antab.replace("= ", "=")
    while "  " in antab: antab = antab.replace("  ", " ")
    if antab[0] == " ": antab = antab[1:]

    # This is a list to be output.
    tables = []

    # read gain
    isgroup=True
    idx0 = 0
    while isgroup:
        # Check if there is a gain table not loaded yet.
        idx0 = antab.find("GAIN", idx0)
        if idx0 < 0:
            isgroup=False
            break

        # Get header
        idx1 = antab.find("/", idx0+1)
        if idx1 < 0:
            raise ValueError("A GAIN table does not have its first delimeter '/'.")
        header = antab[idx0:idx1-1]

        # Get data (only if a GAIN group has TABLE)
        if "TABLE" in header:
            idx2 = antab.find("/", idx1+1)
            if idx2 < 0:
                raise ValueError("A GAIN table does not have its second delimeter '/' for TABLE.")
            data = antab[idx1+1:idx2-1]
        else:
            idx2 = idx1
            data = None

        # Read values
        tables.append(_read_gain(header,data))
        idx0 = idx2+1

    # read tsys
    isgroup=True
    idx0 = 0
    while isgroup:
        # Check if there is a Tsys table not loaded yet.
        idx0 = antab.find("TSYS", idx0)
        if idx0 < 0:
            isgroup=False
            break

        # Get header
        idx1 = antab.find("/", idx0+1)
        if idx1 < 0:
            raise ValueError("A TSYS table does not have its first delimeter '/'.")
        header = antab[idx0:idx1-1]

        # Get data
        idx2 = antab.find("/", idx1+1)
        if idx2 < 0:
            raise ValueError("A TSYS table does not have its second delimeter '/'.")
        data = antab[idx1+1:idx2-1]

        # Read values
        tables.append(_read_tsys(header,data))
        idx0 = idx2+1

    return tables


def _read_gain(header,data):
    '''
    READ GAIN VALUE
    '''
    outdic = {}

    # Initialize
    outdic["GROUP"]=None
    outdic["ANT"]=None
    outdic["DPFU"]=None
    outdic["GCTYPE"]=None
    outdic["SPOL"]=None
    outdic["FREQ"]=None
    outdic["POLY"]=None
    outdic["TABLE"]=None

    # Read header
    keywords = header.split(" ")
    Nkeys = len(keywords)
    outdic["GROUP"] = keywords[0]
    outdic["ANT"] = keywords[1]
    # Check GCTYPE:
    gctypes = ["EQUAT", "ALTAZ", "ELEV", "GCNRAO"]
    isgctype = []
    ntypes=0
    for gctype in gctypes:
        if gctype in keywords:
            outdic["GCTYPE"] = gctype
            ntypes+=1
    if ntypes>1:
        errmsg = "GAIN: Station %s has multiple GC types."%(outdic["ANT"])
        raise ValueError(errmsg)
    elif ntypes==0:
        errmsg = "GAIN: Station %s has no GC type."%(outdic["ANT"])
        raise ValueError(errmsg)
    if outdic["GCTYPE"]=="GCNRAO":
        errmsg = "GAIN: Station %s GC type GCNRAO cannot be loaded in this module."%(outdic["ANT"])
        raise ValueError(errmsg)

    # CHECK DPFU value
    for i in np.arange(2, Nkeys):
        keyword = keywords[i]
        contents = keyword.split("=")
        if contents[0] == "DPFU":
            if outdic["DPFU"] is not None:
                errmsg = "GAIN: Station %s has multiple DPFU keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            minmaxval = np.asarray(contents[1].split(","), dtype=np.float64)
            minmaxval = np.sort(minmaxval)
            outdic["DPFU"] = minmaxval
            break

    # Check other values
    for i in np.arange(2, Nkeys):
        keyword = keywords[i]
        contents = keyword.split("=")
        if   contents[0] == outdic["GCTYPE"] or contents[0] == "DPFU":
            pass
        elif contents[0] == "RCP" or contents[0] == "LCP":
            if outdic["SPOL"] is not None:
                errmsg = "GAIN: Station %s has multiple RCP/LCP keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            if len(outdic["DPFU"]) > 1:
                errmsg = "GAIN: Station %s has two DPFU values for single pol."%(outdic["ANT"])
                raise ValueError(errmsg)
            outdic["SPOL"] = contents[1]
        elif contents[0] == "FREQ":
            if outdic["FREQ"] is not None:
                errmsg = "GAIN: Station %s has multiple FREQ keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            if "," not in contents[1]:
                errmsg = "GAIN: Station %s has an invalid FREQ value."%(outdic["ANT"])
                raise ValueError(errmsg)
            minmaxval = np.asarray(contents[1].split(","), dtype=np.float64)
            minmaxval = np.sort(minmaxval)
            outdic["FREQ"] = minmaxval
        elif contents[0] == "POLY":
            if outdic["POLY"] is not None:
                errmsg = "GAIN: Station %s has multiple POLY keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            if "TABLE" in keywords:
                errmsg = "GAIN: Station %s has both TABLE and POLY keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            outdic["POLY"] = np.asarray(contents[1].split(","), dtype=np.float64)
        else:
            errmsg = "outdic: Station %s has an invalid key %s."%(outdic["ANT"], contents[0])
            raise ValueError(errmsg)

    # Read data
    if "TABLE" in keywords:
        data = data.split(" ")
        while "" in data: data.remove("")

        # check number of data
        Ntot = len(data)
        Ncol = 2
        if Ntot % Ncol != 0:
            errmsg = "GAIN: Station %s has an invalid number of TABLE: mod(Ndata, Ncol=2) != 0"%(outdic["ANT"])
            raise ValueError(errmsg)
        Nrow = Ntot//Ncol

        # Convert data to 2d data
        data = np.asarray(data)
        data = data.reshape([Nrow, Ncol])

        # read data
        table = {}
        names = ["coord", "outdic"]
        for icol in np.arange(Ncol):
            name = names[icol]
            table[name] = np.float64(data[:,icol])
        outdic["TABLE"]=pd.DataFrame(table, columns=names)
    return outdic


def _read_tsys(header,data):
    '''
    READ TSYS VALUE
    '''
    outdic = {}

    # Initialize
    outdic["GROUP"]=None
    outdic["ANT"]=None
    outdic["FT"]=None
    outdic["TIMEOFF"]=None
    outdic["RANGE"]=None
    outdic["INDEX"]=None

    # Read header
    keywords = header.split(" ")
    Nkeys = len(keywords)
    outdic["GROUP"] = keywords[0]
    outdic["ANT"] = keywords[1]
    for i in np.arange(2, Nkeys):
        keyword = keywords[i]
        contents = keyword.split("=")
        # Read each keyword
        if   contents[0] == "FT":
            if outdic["FT"] is not None:
                errmsg = "TSYS: Station %s has multiple FT keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            outdic["FT"] = np.float64(contents[1])
        elif contents[0] == "TIMEOFF":
            if outdic["TIMEOFF"] is not None:
                errmsg = "TSYS: Station %s has multiple TIMEOFF keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            outdic["TIMEOFF"] = np.float64(contents[1])
        elif contents[0] == "RANGE":
            if outdic["RANGE"] is not None:
                errmsg = "TSYS: Station %s has multiple RANGE keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            if "," not in contents[1]:
                errmsg = "TSYS: Station %s has an invalid RANGE value."%(outdic["ANT"])
                raise ValueError(errmsg)
            minmaxval = np.asarray(contents[1].split(","), dtype=np.float64)
            minmaxval = np.sort(minmaxval)
            outdic["RANGE"] = minmaxval
        elif contents[0] == "INDEX":
            if outdic["INDEX"] is not None:
                errmsg = "TSYS: Station %s has multiple INDEX keys."%(outdic["ANT"])
                raise ValueError(errmsg)
            outdic["INDEX"] = contents[1].replace("'", "").split(",")
        else:
            errmsg = "TSYS: Station %s has an invalid key %s."%(outdic["ANT"], contents[0])
            raise ValueError(errmsg)

    # CHECK INDEX
    if outdic["INDEX"] is None:
        outdic["INDEX"] = ["NOINDEX"]
        Nidx = 1
    else:
        Nidx = len(outdic["INDEX"])

    # Read data
    data = data.split(" ")
    while "" in data: data.remove("")

    # check number of data
    Ntot = len(data)
    Ncol = Nidx+2
    if Ntot % Ncol != 0:
        errmsg = "TSYS: Station %s has an invalid number of data: mod(Ndata, Ncol=Nindex+2) != 0"%(outdic["ANT"])
        raise ValueError(errmsg)
    Nrow = Ntot//Ncol

    # Convert data to 2d data
    data = np.asarray(data)
    data = data.reshape([Nrow, Ncol])
    #print(data)

    # read data
    table = {}
    names = ["DOY", "TIME"] + outdic["INDEX"]
    for icol in np.arange(Ncol):
        name = names[icol]
        if name == "TIME":
            table[name] = data[:,icol]
        else:
            table[name] = np.float64(data[:,icol])
    outdic["DATA"]=pd.DataFrame(table, columns=names)
    return outdic

# ------------------------------------------------------------------------------
# write to an ANTAB file
# ------------------------------------------------------------------------------
def write_antab(antabdata, filename=None):
    lines = []
    for curdata in antabdata:
        if curdata["GROUP"] == "GAIN":
            lines += _write_gain(curdata)
        elif curdata["GROUP"] == "TSYS":
            lines += _write_tsys(curdata)
    lines="".join(lines)
    if filename is None:
        return lines
    else:
        f=open(filename, "w")
        f.write(lines)
        f.close()

def _write_tsys(data, maxchar=80):
    lines = []
    line="%s %s"%(data["GROUP"], data["ANT"])

    # Factor
    if data["FT"] is not None:
        string = "FT=%e"%(data["FT"])
        line, lines = _add_str(line, lines, string, maxchar)

    # TIMEOFF
    if data["TIMEOFF"] is not None:
        string = "TIMEOFF=%e"%(data["TIMEOFF"])
        line, lines = _add_str(line, lines, string, maxchar)

    # RANGE
    if data["RANGE"] is not None:
        if len(data["RANGE"])!=2:
            raise ValueError("Invalid number of RANGE Values.")
        string = "RANGE=%e,%e"%(data["RANGE"][0],data["RANGE"][1])
        line, lines = _add_str(line, lines, string, maxchar)

    # INDEX
    if data["INDEX"][0] != "NOINDEX":
        string = "INDEX='"
        string += "','".join(data["INDEX"])
        string += "'"
        line, lines = _add_str(line, lines, string, maxchar)
    line, lines = _app_line(line, lines)

    lines.append("/\n")

    Nrow = data["DATA"].shape[0]
    Ncol = len(data["INDEX"])+2
    columns = ["DOY","TIME"]+data["INDEX"]
    for irow in np.arange(Nrow):
        string = "%3d "%(data["DATA"].loc[irow,"DOY"])
        string+= "%11s "%(data["DATA"].loc[irow,"TIME"])
        for icol in np.arange(2,Ncol):
            string+= "%e "%(data["DATA"].loc[irow,columns[icol]])
        lines.append(string[:-1]+"\n")
    lines.append("/\n")
    return lines

def _write_gain(data, maxchar=80):
    lines = []
    line="%s %s %s"%(data["GROUP"], data["ANT"], data["GCTYPE"])

    # Polarization
    if data["SPOL"] is not None:
        string = data["SPOL"]
        line, lines = _add_str(line, lines, string, maxchar)

    # DPFU
    string = "DPFU=%e"%(data["DPFU"][0])
    if len(data["DPFU"])==2:
        string += ",%e"%(data["DPFU"][1])
    elif len(data["DPFU"])>2:
        raise ValueError("Invalid number of DPFU Values.")
    line, lines = _add_str(line, lines, string, maxchar)

    # FREQ
    if data["FREQ"] is not None:
        if len(data["FREQ"])!=2:
            raise ValueError("Invalid number of FREQ Values.")
        string = "FREQ=%e,%e"%(data["FREQ"][0],data["FREQ"][1])
        line, lines = _add_str(line, lines, string, maxchar)

    if data["POLY"] is not None:
        string = "POLY=%e"%(data["POLY"][0])
        if len(data["POLY"])>1:
            for i in np.arange(1,len(data["POLY"])):
                string += ",%e"%(data["POLY"][i])
        line, lines = _add_str(line, lines, string, maxchar)

    if data["TABLE"] is not None:
        string = "TABLE"
        line, lines = _add_str(line, lines, string, maxchar)
        line, lines = _app_line(line, lines)
        lines.append("/\n")
        Nrow = data["TABLE"].shape[0]
        for irow in np.arange(Nrow):
            coord = data["TABLE"].loc[irow, "coord"]
            gainv = data["TABLE"].loc[irow, "gain"]
            string = "%e %e\n"%([coord, gainv])
            lines.append(string)
    else:
        line, lines = _app_line(line, lines)
    lines.append("/\n")
    return lines

def _add_str(line, lines, string, maxchar=80):
    if len(line)+1+len(string) > maxchar:
        line, lines = _app_line(line, lines)
        line = string
    else:
        line+= " "+string
    return line, lines

def _app_line(line, lines):
    lines.append(line+"\n")
    line = ""
    return line, lines
