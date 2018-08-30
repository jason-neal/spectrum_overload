# spectrum_overload
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/jason-neal)
[![PyPI version](https://badge.fury.io/py/spectrum-overload.svg)](https://badge.fury.io/py/spectrum-overload)[![Updates](https://pyup.io/repos/github/jason-neal/spectrum_overload/shield.svg)](https://pyup.io/repos/github/jason-neal/spectrum_overload/)[![Python 3](https://pyup.io/repos/github/jason-neal/spectrum_overload/python-3-shield.svg)](https://pyup.io/repos/github/jason-neal/spectrum_overload/)

[![Documentation Status](https://readthedocs.org/projects/spectrum-overload/badge/?version=latest)](http://spectrum-overload.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/jason-neal/spectrum_overload.svg?branch=master)](https://travis-ci.org/jason-neal/spectrum_overload) [![Coverage Status](https://coveralls.io/repos/github/jason-neal/spectrum_overload/badge.svg?branch=master)](https://coveralls.io/github/jason-neal/spectrum_overload?branch=master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3e9a2cf4ad914e0ebc24b6b2b83059db)](https://www.codacy.com/app/jason-neal/spectrum_overload?utm_source=github.com&utm_medium=referral&utm_content=jason-neal/spectrum_overload&utm_campaign=badger)[![Test Coverage](https://codeclimate.com/github/jason-neal/spectrum_overload/badges/coverage.svg)](https://codeclimate.com/github/jason-neal/spectrum_overload/coverage) [![Code Climate](https://codeclimate.com/github/jason-neal/spectrum_overload/badges/gpa.svg)](https://codeclimate.com/github/jason-neal/spectrum_overload)[![Issue Count](https://codeclimate.com/github/jason-neal/spectrum_overload/badges/issue_count.svg)](https://codeclimate.com/github/jason-neal/spectrum_overload)  


## Overview
The purpose of this was project was to learn how to use and create Classes, and to create a Spectrum object to use within my Phd work.

The main goals of this project are basically complete.

- create a class to contain spectrum
- automatic interpolation
- overloaded operators
especially
- Spectral division (spectrum_A / spectrum_B )
- Spectral subtraction (spectrum_A - spectrum_B)
- Powers/exponents (spectrum ** x)

Further documentation can be found on [read the docs](https://spectrum-overload.readthedocs.io/en/latest/).

##### Note:
    When the spectra have different wavelength vectors spectrum_overload interpolates
    to the wavelength of the first spectrum. This may not suit your requirements.


## Installation
### Pip

    pip install spectrum-overload

### git / manual
Currently to install and use Spectrum class for your own projects.

clone the git repository where you want it then type:

    git clone https://github.com/jason-neal/spectrum_overload.git
    cd spectrum_overload
    python setup.py install

or `python setup.py develop` if you want to make changes.


## Usage
To use import the class using :

    from spectrum_overload import Spectrum
    ...
    my_spectrum = Spectrum(flux, xaxis)

or :

    from spectrum_overload.spectrum import Spectrum as spec
    ...
    my_spectrum = spec(flux, xaxis)

or however else you would like to name it.

A tutorial is provided [here](Notebooks/Tutorial.ipynb) to show an example of how to use this class.


## Contributions
Contributions are very welcome.

I would really appreciate user feedback or suggested improvements if you have any.

Feel free to submit issues or create pull requests.


## Python 2.7
Python 2.7 is only supported in versions <0.2.1 (also available on pip) to make use of useful python3 features.


## Other Spectrum Classes

There are many other spectrum classes around but I didn't see any that overload the operators. (I could be blind).

One of these may better suit your needs

- https://github.com/crawfordsm/specreduce
- https://github.com/crawfordsm/pyspectrograph/tree/master/PySpectrograph/Spectra
- http://pyspeckit.bitbucket.org/html/sphinx/spectrum.html#pyspeckit.spectrum.classes.Spectrum
- https://github.com/cokelaer/spectrum
- https://github.com/astropy/specutils

Wow A lot. I probably should not reinvent the wheel too much then...

It turns out that astropy/specutils is very similar to what I have created but its direction is uncertain at the moment and they do not use overloaded operators and will not implement in the foreseeable future.

## TO DO
Some tasks still to do:
- Improve Documentation
- Generate Calibration solution (outside spectrum class)?
