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
        # Some checks before creating class
        if isinstance(flux, str):
            raise TypeError("Cannot assign {} to the flux attribute".format(type(flux)))
        elif isinstance(xaxis, str):
            raise TypeError("Cannot assign {} to the xaxis attribute".format(type(xaxis)))
        
        if flux is not None:
            self._flux = np.asarray(flux)
        else:
            self._flux = flux

        if xaxis is None and flux is not None:
            # Applying range to xaxis of equal length as flux 
            try:
                # Try to assign arange the length of flux
                self._xaxis = np.arange(len(flux))
            except TypeError: 
                self._xaxis = None
        else:
            self._xaxis = np.asarray(xaxis) # Still need asarray as setter is not used here
        
        # Check assigned lenghts
        self.length_check() 
        self.calibrated = calibrated
        self.header = header   # Access header with a dictionary call.
    
    @property
    def xaxis(self):
        #print("Getting xaxis property")
        return self._xaxis 

    @xaxis.setter 
    def xaxis(self, value):
        #print("xaxis value = ", value)
        if isinstance(value, str): 
            #Try to catch some bad assignments 
            # Yes a list of strings will not be caught
            raise TypeError("Cannot assign {} to the xaxis attribute".format(type(value)))
        elif value is None:
            try:
                # Try to assign arange the length of flux
                self._xaxis = np.arange(len(self._flux))
            except TypeError: 
                # if self._flux is None then it has no length.
                self._xaxis = None
            #print("assigning xaxis the same length of _flux")
            
        # Add any other checks in here if nessary
        elif self._flux is not None:
            if len(value) != len(self._flux):
                raise ValueError("Lenght of xaxis does not match the length of flux ")
            else:
                self._xaxis = np.asarray(value)
    
    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, value):
        if isinstance(value, str): 
            #Try to catch some bad assignments 
            # Yes a list of strings will not be caught
                raise TypeError("Cannot assign {} to the flux attribute".format(type(value)))

        if value is not None:
            print("Turning flux input into np array")
            #Not checking to make sure it equals the xaxis
            # If changing flux and xaxis set the flux first
            self._flux = np.asarray(value)
        else:
            self._flux = value 

    def length_check(self):
        """ Check length of xaxis and flux are equal.
        Raise error if they are not 
        If everyting is ok then there is no response/output"""
        if (self._flux is None) and (self._xaxis is None):
            # Can't measure lenght of none
            pass
        elif (self._flux is None) or (self._xaxis is None):
            pass
        elif len(self._flux) != len(self._xaxis):
            raise ValueError("The length of xaxis and flux must be the same")


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
                self.flux = self.flux[mask]    # change flux first
                self.xaxis = self.xaxis[mask]
            except TypeError:
                print("Make sure your xaxis is an array")
                #Return to original values
                self.flux = flux_org           # change flux first
                self.xaxis = x_org
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


    def __truediv__(self, other):
        """Overload truedivision (/) to divide two specta """
        if isinstance(other, Spectrum):
            if self.calibrated != other.calibrated:
                """Checking the Spectra are of same calibration state"""
                raise SpectrumError("The Spectra are not of the same calibration state.")
        
            if np.all(self.xaxis == other.xaxis):
                # Easiest condition in which xaxis of both are the same
                try:
                    new_flux = self.flux / other.flux
                except ZeroDivisionError:
                    print("Some of the spectrum was zero. Replacing with Nans")
                    nand_other = other.flux
                    nand_other[nand_other == 0] = np.nan()
                    new_flux = self.flux / other.flux
           
        else:
            new_flux = self.flux / other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, calibrated=self.calibrated)
    
    def __add__(self, other):
        if isinstance(other, Spectrum):
            if self.calibrated != other.calibrated:
                """Checking the Spectra are of same calibration state"""
                raise SpectrumError("The Spectra are not of the same calibration state.")
            
            if np.all(self.xaxis == other.xaxis):
                # Easiest condition in which xaxis of both are the same
                new_flux = self.flux + other.flux
        else:
            new_flux = self.flux + other
        
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)

    def __radd__(self, other):
        # E.g. for first Item in Sum  0  + Spectrum fails.
        
        new_flux = self.flux + other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)

    
    def __sub__(self, other):
        if isinstance(other, Spectrum):
            if self.calibrated != other.calibrated:
                """Checking the Spectra are of same calibration state"""
                raise SpectrumError("The Spectra are not of the same calibration state.")
            # Only for equal xaxis
            if np.all(self.xaxis == other.xaxis):
                # Easiest condition in which xaxis of both are the same
                new_flux = self.flux - other.flux
        else:
            new_flux = self.flux - other

        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)


    def __mul__(self, other):
        if isinstance(other, Spectrum):
            if self.calibrated != other.calibrated:
                """Checking the Spectra are of same calibration state"""
                raise SpectrumError("The Spectra are not of the same calibration state.")
            # Only for equal xaxis
            if np.all(self.xaxis == other.xaxis):
                # Easiest condition in which xaxis of both are the same
                new_flux = self.flux * other.flux
        else:
            new_flux = self.flux * other

        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)


    def __pow__ (self, other):
        # Overlaod to use power to scale the flux of the spectra
        #if len(other) > 1 :
        #    raise ValueError("Spectrum can only be raised to the power of one number not {}".format(len(other)))
        try:
            new_flux = self.flux ** other
            return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header, calibrated=self.calibrated)
        
        except :
            #Tpye error or value error are likely
            raise 


    def __len__(self):
        """ Return length of flux Spectrum"""
        return len(self.flux)

    def __neg__ (self):
        """ Take negative flux """
        negflux = -self.flux
        return Spectrum(flux=negflux, xaxis=self.xaxis, calibrated=self.calibrated, header=self.header)
      
    def __pos__ (self):
        """ Take positive flux """
        posflux = +self.flux
        return Spectrum(flux=posflux, xaxis=self.xaxis, calibrated=self.calibrated, header=self.header)
  
    def __abs__ (self):
        """ Take absolute flux """
        absflux = abs(self.flux)
        return Spectrum(flux=absflux, xaxis=self.xaxis, calibrated=self.calibrated, header=self.header)
  

## TO DO !
#--------------------
# Add an interpolation
# Incorporate interpolation into all overloaded operators

# Setter /getter for xaxis and flux to turn into np.asarrays





class SpectrumError(Exception):
    pass