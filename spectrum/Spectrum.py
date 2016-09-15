#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
# Spectrum Class

# Begun August 2016
# Jason Neal


class Spectrum(object):
    """ Spectrum class represents and manipulates astronomical spectra. """
    
    def __init__(self, flux=None, xaxis=None, header=None, calibrated=False):
        """ Initalise a Spectrum object """
        self.xaxis = np.asarray(xaxis)
        self.flux = np.asarray(flux)
        self.calibrated = calibrated
        self.header = header   # Access header with a dictionary call.

    def wav_select(self, wav_min, wav_max):
        """ Select the spectrum between wav_min and wav_max values 
            Uses numpy slicing for high speed.
        """
        x_org = self.xaxis
        flux_org = self.flux
        if len(self.xaxis) == 0:
            print("No xaxis to select from")
        else:
            try:
                mask = (self.xaxis > wav_min) & (self.xaxis < wav_max)
                self.xaxis = self.xaxis[mask]
                self.flux = self.flux[mask]
            except TypeError:
                print("Make sure your xaxis is an array")
                #Return to original values
                self.xaxis = x_org
                self.flux = flux_org
                raise


    def doppler_shift(self, RV):
        ''' Function to compute a wavelenght shift due to radial velocity
        using RV / c = delta_lambda/lambda
        RV - radial velocity (in km/s)
        lambda_rest - rest wavelenght of the spectral line
        delta_lambda - (lambda_final - lambda_rest)
        '''
        if abs(RV) < 1e-7:
            """ RV smaller then 0.1 mm/s"""
            print("Warning the RV value given is very small (<0.1 mm/s).\n " 
                  "Not performing the doppler shift")

        elif np.isnan(RV) or np.isinf(RV):
            print("Warning RV is infinity or Nan. "
                  "Not performing the doppler shift")

        elif self.calibrated:
            c = 299792.458
            lambdaShift = self.xaxis * (RV / c)
            self.xaxis = self.xaxis + lambdaShift
        else:
            print("Attribute xaxis is not wavelength calibrated. Cannot perform doppler shift")


    def calibrate_with(self, wl_map):
        """ Calibrate with polynomial with parameters wl_map.
        Input:
            wl_map - Polynomial cooeficients that take the form expected by np.poylval()
        Output:
            self.xaxis is replaced with the calibrated spectrum
            self.calibrated is set to True
        The parameters can be generated by np.polyfit(x, y, order)
        """
        if self.calibrated:
            print("Spectrum already calibrated, Not Calibrating again.")
        else:
            wavelength = np.polyval(wl_map, self.xaxis)   # Polynomail parameters
            self.xaxis = wavelength
            self.calibrated = True  # Set calibrated Flag 
        
        if np.any(self.xaxis <= 0):
            print("Warning! The wavlength solution contains a value of zero. "
                  "Please check your calibrations\nThis will not doppler "
                  "shift correctly. This may raise an error in the future.")


    def interpolate_to(self, spectrum):
        """Interpolate wavelength solution to wavelength of spectrum
        Think about weather this should be spectrum or sepctrum.xaxis (just the wavelength) 
        
        A comment from ENAA 2016 regarded interpolation. 
        Interpolation techniques need to be tested to acheive best
        performance for low signal applications. i.e. direct exoplanet detection"""
        pass


    #######################################################
    #Overloading Operators
    #######################################################


    def ___truediv___(self, other):
        """Overload truedivision (/) to divide two specta """
        if self.calibrated != other.calibrated:
            """Checking the Spectra are of same calibration state"""
            raise SpectrumCalibrationError("The Spectra are not of the same calibration state.")
        
        if np.all(self.xaxis == other.xaxis):
            # Easiest condition in which xaxis of both are the same
            new_flux = self.flux / other.flux
            return Spectrum(flux=new_flux, xaxis=self.xaxis, calibrated=self.calibrated)

    
    def __add__(self, other):
        if self.calibrated != other.calibrated:
            """Checking the Spectra are of same calibration state"""
            raise SpectrumCalibrationError("The Spectra are not of the same calibration state.")
        
        if np.all(self.xaxis == other.xaxis):
            # Easiest condition in which xaxis of both are the same
            new_flux = self.flux + other.flux
            return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)

    def __radd__(self, other):
        # E.g. for first Item in Sum  0  + Spectrum fails.
        
        new_flux = self.flux + other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)



    def __mul__(self, other):
        if self.calibrated != other.calibrated:
            """Checking the Spectra are of same calibration state"""
            raise SpectrumCalibrationError("The Spectra are not of the same calibration state.")
        # Only for equal xaxis
        if np.all(self.xaxis == other.xaxis):
            # Easiest condition in which xaxis of both are the same
            new_flux = self.flux * other.flux
            return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)


    def __pow__ (self, other):  # [, modulo]  extra parameter to be able to use pow() function
        # Overlaod to use power to scale the flux of the spectra
        if len(other) > 1 :
            raise ValueError("Spectrum can only be raised to the power of one number not {}".format(len(other)))
        try:
            new_flux = self.flux ** other
            return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)
        except :
            #Tpye error or value error are likely
            raise 


    def __len__(self):
        # Give the length of the spectrum:
        return len(Spectrum.flux)

## TO DO !
#--------------------
# Overrideopperators such 
# e.g, plus, minus, subtract, divide

# Interpolation in wavelength (before subtraction)
