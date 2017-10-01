from setuptools import setup, find_packages
from codecs     import open
from os         import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README"), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="eat",
    version="0.9.0",
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
)
