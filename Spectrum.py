#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
# Spectrum Class

# Begun August 2016
# Jason Neal


class Spectrum:
    """ Spectrum class represents and manipulates astronomical spectra. """
    
    def __init__(self, flux=[], xaxis=[], calibrated=False):
        """ Create a empty spectra """
        self.xaxis = np.asarray(xaxis)
        self.flux = np.asarray(flux)
        self.calibrated = calibrated

    def wav_select(self, wav_min, wav_max):
        """ Fast Wavelength selector between wav_min and wav_max values 
        If passed lists it will return lists.
        If passed np arrays it will return arrays
    
        """
        try:
            mask = (self.xaxis > wav_min) & (self.xaxis < wav_max)
            wav_sel = self.xaxis[mask]
            flux_sel = self.flux[mask]
        except TypeError:
            print("Make sure your xaxis is an array")
              raise
        # Set new spectra
        self.xaxis = wav_sel
        self.flux = flux_sel


# Try using Spectrum 
#if __name__ == __main__:

