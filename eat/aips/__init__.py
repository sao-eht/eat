#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
A python module eat.aips

This is a submodule of eat for running AIPS in python.
'''
def setenv(
        aipsdir="/usr/local/aips",
        aipsver=None,
        ptdir="/usr/share/parseltongue/python",
        obitdir="/usr/lib/obit/python"):
    '''
    This function sets PYTHONPATHs for ParselTongue

    Args:
      aipsdir (str):
        Your AIPS direcory. This parameter will be used only when
        AIPS_DIR/LOGIN.SH is not loaded on your Shell.
        (e.g., default) /usr/local/aips

      aipsver (str; default=None (latest version); e.g. 31DEC17):
        Your AIPS version. If this is none, it will use your default version in
        LOGIN.SH.

      ptdir (str):
        pythonpath for ParselTongue. This should end by "/python"
        (e.g., default) /usr/share/parseltongue/python

      obitdir (str):
        pythonpath for Obit. This should end by "/python"
        (e.g., default) /usr/lib/obit/python
    '''
    import sys
    import os

    # LOAD LOGIN.SH
    print("Check if $AIPS_ROOT/LOGIN.SH is already loaded:")
    if "AIPS_VERSION" in os.environ and "AIPS_ROOT" in os.environ:
        print("  Yes.")
        loginsh = os.path.join(os.environ["AIPS_ROOT"],"LOGIN.SH")
    else:
        print("  No: LOGIN.SH is not loaded.")
        loginsh = os.path.join(aipsdir,"LOGIN.SH")
        _source(loginsh)

    # AIPSDIR
    aipsdir_org = os.environ["AIPS_ROOT"]
    print(("  AIPS directory: %s"%(aipsdir_org)))

    # CHECK DEFAULT AIPS VERISON
    aipsver_org = os.path.split(os.environ["AIPS_VERSION"])[1]
    if aipsver is None:
        aipsver = aipsver_org

    print(("  Default   AIPS version: %s"%(aipsver_org)))
    print(("  Specified AIPS version: %s"%(aipsver.upper())))

    if aipsver_org == aipsver.upper():
        print("  > No change in AIPS version.")
        aipsver_new = aipsver_org

    else:
        if os.path.isdir(os.path.join(aipsdir_org, aipsver.upper())):
            aipsver_new = aipsver.upper()
            print(("  > AIPS version changed to %s."%(aipsver.upper())))
        else:
            errmsg = "AIPS Version %s is not installed in %s."%(aipsver.upper(), aipsdir_org)
            raise ValueError(errmsg)

    # CHECK OTHER SCRIPTS
    prdevssh = os.path.join(os.environ["AIPS_ROOT"], "PRDEVS.SH")
    prdevssh = os.path.join(aipsdir_org, os.readlink(prdevssh))
    prdevssh = prdevssh.replace(aipsver_org, aipsver_new)
    print(("  PRDEVS.SH Location: %s"%(prdevssh)))

    dadevssh = prdevssh.replace("PRDEVS.SH", "DADEVS.SH")
    print(("  DADEVS.SH Location: %s"%(dadevssh)))

    # Source Bash files
    _source(loginsh, replace=[aipsver_org, aipsver_new])
    _source(prdevssh)
    _source(dadevssh)

    if os.path.isdir(ptdir):
        print(("ParselTongue Python DIR: %s"%(ptdir)))
    else:
        errmsg="ParselTongue Python Directory '%s' is not available. "%(ptdir)
        errmsg+="Please set PYTHONPATH with eat.aips.setenv() first."
        raise ValueError(errmsg)

    if os.path.isdir(obitdir):
        print(("Obit Python DIR: %s"%(obitdir)))
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
            print(("ParselTongue Version: "+ptversion.version))
    except ImportError:
        print("[Error] PYTHONPATHs to ParselTongue and/or Obit are not correctly set.")
        print("        Please set PYTHONPATHs with eat.aips.setenv().")
        raise
    except:
        print("[Error] Internal Error in ParselTongue: import ptversion")
        raise


def _source(script, replace=None, update=True):
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
        print(("  Reading Environmental Variables from %s"%(script)))

    pipe = Popen(". %s > /dev/null 2>&1; env" % script, stdout=PIPE, shell=True)
    data = str(pipe.communicate()[0])
    env = dict((line.split("=", 1) for line in data.splitlines()))
    if replace is not None:
        for key in list(env.keys()):
            value = env[key]
            if replace[0] in value:
                env[key] = env[key].replace(replace[0], replace[1])

    if update:
        environ.update(env)
    else:
        return env
