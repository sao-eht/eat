#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A python module eat.aips

This is a submodule of eat for running AIPS in python.
'''
def setenv(
        aipsdir="/usr/local/aips",
        ptdir="/usr/share/parseltongue/python",
        obitdir="/usr/lib/obit/python"):
    '''
    This function sets PYTHONPATHs for ParselTongue
    
    Args:
      aipsdir (str):
        Your AIPS direcory. This parameter will be used only when
        AIPS_DIR/LOGIN.SH is not loaded on your Shell.
        (e.g., default) /usr/local/aips
        
      ptdir (str):
        pythonpath for ParselTongue. This should end by "/python"
        (e.g., default) /usr/share/parseltongue/python
    
      obitdir (str):
        pythonpath for Obit. This should end by "/python"
        (e.g., default) /usr/lib/obit/python
    '''
    import sys
    import os
    
    print("Check if $AIPS_ROOT/LOGIN.SH is already loaded:")
    if os.environ.has_key("AIPS_VERSION") and os.environ.has_key("AIPS_ROOT"):
        print("  Yes.")
    else:
        print("  No: LOGIN.SH is not loaded.")
        sourcefile = os.path.join(aipsdir,"LOGIN.SH")
        _source(sourcefile)
    
    sourcefile = os.path.join(os.environ["AIPS_ROOT"], "PRDEVS.SH")
    _source(sourcefile)
    
    sourcefile = os.path.join(
        os.path.split(sourcefile)[0],
        os.path.split(os.readlink(sourcefile))[0],
        "DADEVS.SH"
    )
    _source(sourcefile)
    
    if os.path.isdir(ptdir):
        print("ParselTongue Python DIR: %s"%(ptdir))
    else:
        errmsg="ParselTongue Python Directory '%s' is not available. "%(ptdir)
        errmsg+="Please set PYTHONPATH with eat.aips.setenv() first."
        raise ValueError(errmsg)
    
    if os.path.isdir(obitdir):
        print("Obit Python DIR: %s"%(obitdir))
    else:
        errmsg="Obit Python Directory '%s' is not available. "%(obitdir)
        errmsg+="Please set PYTHONPATH with eat.aips.setenv() first."
        raise ValueError(errmsg)
    
    if ptdir not in sys.path:
        sys.path.insert(0,ptdir)
    
    if obitdir not in sys.path:
        sys.path.insert(0,obitdir)
    
    check()


def check(printver=True):
    '''
    This function checks the version of ParselTongue.
    '''
    try:
        import ptversion
        if printver:
            print("ParselTongue Version: "+ptversion.version)
    except ImportError:
        print("[Error] PYTHONPATHs to ParselTongue and/or Obit are not correctly set.")
        print("        Please set PYTHONPATHs with eat.aips.setenv().")
        raise
    except:
        print("[Error] Internal Error in ParselTongue: import ptversion")
        raise


def _source(script, update=True):
    """
    Source variables from a shell script
    import them in the environment (if update==True)
    """
    from subprocess import Popen, PIPE
    from os import environ
    import os
    
    if os.path.isfile(script) is False:
        errmsg = "'%s' is not available"%(script)
        raise ValueError(errmsg)
    else:
        print("Reading Envrioment Variables from %s"%(script))
    
    pipe = Popen(". %s > /dev/null 2>&1; env" % script, stdout=PIPE, shell=True)
    data = pipe.communicate()[0]
    env = dict((line.split("=", 1) for line in data.splitlines()))

    if update:
        environ.update(env)
    else:
        return env
