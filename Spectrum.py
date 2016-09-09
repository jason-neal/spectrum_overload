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

    def doppler_shift(self, RV):
        ''' Function to compute a wavelenght shift due to radial velocity
        using RV / c = delta_lambda/lambda
        RV - radial velocity (in km/s)
        lambda_rest - rest wavelenght of the spectral line
        delta_lambda - (lambda_final - lambda_rest)
        '''
        if self.calibrated:
            c = 299792.458
            lambdaShift = self.xaxis * (RV / c)
            self.xaxis = self.xaxis + lambdaShift
        else:
            print("Attribute xaxis is not wavelength calibrated. Cannot perform doppler shift")
# Try using Spectrum 
#if __name__ == __main__:

