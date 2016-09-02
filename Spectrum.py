#!/usr/bin/python
from __future__ import print_function, division

# Spectrum Class

# Begun August 2016
# Jason Neal


class Spectrum:
    """ Spectrum class represents and manipulates astronomical spectra. """
    
    def __init__(self, pixel=[], flux=[], wavelength=[]):
        """ Create a empty spectra """
        self.pixel = pixel
        self.flux = flux
        self.wavelength = wavelength




# Try using Spectrum 
#if __name__ == __main__:

