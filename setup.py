from setuptools import setup, find_packages
from codecs     import open
from os         import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="eat",
    version="1.6",
    description="EHT Analysis Toolkit",
    long_description=long_description,

    url="https://github.com/sao-eht/eat",
    author="Lindy Blackburn",
    author_email="lblackburn@cfa.harvard.edu",

    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python",
    ],
    keywords="astronomy analysis",

    packages=find_packages(exclude=["doc*", "test*"]),
    package_data={'eat': ["data/*.csv"]},

    scripts=["bin/addflag.py",
             "bin/adjustres.py",
             "bin/alistadhoc",
             "bin/alma_adhoc",
             "bin/alma_delayoffs",
             "bin/alma_fixsqrt2",
             "bin/alma_pcal",
             "bin/alma_sbdmbd",
             "bin/antab2sefd",
             "bin/applycal",
             "bin/cal_apriori_pang_uvfits.py",
             "bin/cal_polcal_gains_uvfits.py",
             "bin/caluvfits.py",
             "bin/closecf",
             "bin/fitsidx",
             "bin/fix_lopcal",
             "bin/fplotpdf",
             "bin/gainratiocal",
             "bin/generate_polcal_table.py",
             "bin/hops2uvfits.py",
             "bin/import_uvfits.py",
             "bin/mksnr",
             "bin/noauto",
             "bin/ovex2codes",
             "bin/printalist",
             "bin/runall.py",
             "bin/uv_comb.py"]
)
