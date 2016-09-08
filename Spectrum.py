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
        if isinstance(self.xaxis, list): # If passed lists
            wav_sel = [wav_val for wav_val in self.xaxis if (wav_min < wav_val < wav_max)]
            flux_sel = [flux_val for wav_val, flux_val in zip(self.xaxis, self.flux) if (wav_min < wav_val < wav_max)]
        elif isinstance(self.xaxis, np.ndarray):
            # Super Fast masking with numpy
            mask = (self.xaxis > wav_min) & (self.xaxis < wav_max)
            wav_sel = self.xaxis[mask]
            flux_sel = self.flux[mask]
        
        else:
              raise TypeError("Unsupported input wav type of type ", type(self.xaxis))
        # Set new spectra
        self.xaxis = wav_sel
        self.flux = flux_sel


# Try using Spectrum 
#if __name__ == __main__:

