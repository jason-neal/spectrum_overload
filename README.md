# spectrum_overload [![Build Status](https://travis-ci.org/jason-neal/spectrum_overload.svg)](https://travis-ci.org/jason-neal/spectrum_overload) [![Coverage Status](https://coveralls.io/repos/github/jason-neal/spectrum_overload/badge.svg?branch=develop)](https://coveralls.io/github/jason-neal/spectrum_overload?branch=develop)

## Overview
The purpose of this was project was to learn how to use and create Classes, and to create a Spectrum object to use within my Phd work.

The main goals of this project are basically complete.

- create a class to contain spectrum
- automatic interpolation 
- overloaded operators
especially
- Spectral division (SpectrumA / SpectrumB )
- Spectral subtraction (SpectrumA - SpectrumB)
- Powers/exponents (Spectrum ** x)

## Installation
Currently to install and use Spectrum class for your own projects.

clone the git repository where you want it:
  
    https://github.com/jason-neal/spectrum_overload.git

cd into the downloaded directory:

    cd spectrum_overload

and install using:

    python setup.py install


## Usage
To use import the class using :
    
    from spectrum_overload import Spectrum
    ...
    my_spectrum = Spectrum.Spectrum(flux, xaxis)

or :

    from spectrum_overload.Spectrum import Spectrum as spec
    ...
    my_spectrum = spec(flux, xaxis)

or how ever else you would like to import it.

A tutorial is provided [here](Notebooks/Tutorial.ipynb) to show an example of how to use this class.


## Contributions
Contributions are very welcome.

I would really appreciate user feedback or suggested improvements if you have any.

Feel free to submit issues or create pull requests.



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
- Push a version to pypi

