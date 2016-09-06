#!/usr/bin/python
from __future__ import print_function, division

# Spectrum Class

# Begun August 2016
# Jason Neal


class Spectrum:
    """ Spectrum class represents and manipulates astronomical spectra. """
    
    def __init__(self, flux=[], xaxis=[], calibrated=False):
        """ Create a empty spectra """
        self.xaxis = xaxis
        self.flux = flux
        self.calibrated = calibrated




# Try using Spectrum 
#if __name__ == __main__:

