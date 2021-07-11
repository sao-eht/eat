# The Event Horizon Telescope Analysis Toolkit

## Dependencies

* python 2.7+
* numpy
* scipy
* matplotlib
* pandas
* astropy
* HOPS (for mk4 access, optional)
* seaborn (for some data inspection plots, optional)

## Installation

`eat` can be used directly by adding the module location to your Python path,
```
git clone https://github.com/sao-eht/eat.git
python
>>> import sys
>>> sys.path.append('eat')
>>> import eat
```

Alternatively for a system install,
```
git clone https://github.com/sao-eht/eat.git
pip install -e eat
```

## Documentation

[Sphinx apidoc](https://eat.readthedocs.io/en/latest/).

## HOPSTOOLS

Tools to convert data from the HOPS format.
