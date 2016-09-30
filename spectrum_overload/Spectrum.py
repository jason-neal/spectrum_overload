#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
import copy
from scipy.interpolate import interp1d
# Spectrum Class

# Begun August 2016
# Jason Neal


class Spectrum(object):
    """ Spectrum class represents and manipulates astronomical spectra. """

    def __init__(self, flux=None, xaxis=None, calibrated=False, header=None):
        """ Initalise a Spectrum object """
        # Some checks before creating class
        if isinstance(flux, str):
            raise TypeError("Cannot assign {} to the flux attribute".format(
                type(flux)))
        elif isinstance(xaxis, str):
            raise TypeError("Cannot assign {} to the xaxis attribute".format(
                type(xaxis)))

        if flux is not None:
            self._flux = np.asarray(flux)
        else:
            self._flux = flux

        if xaxis is None:
            if flux is None:
                self._xaxis = None
            else:
                # Applying range to xaxis of equal length of flux
                try:
                    self._xaxis = np.arange(len(flux))
                except TypeError:
                    print("TypeError caught becasue flux has no length")
                    self._xaxis = None
        else:
            self._xaxis = np.asarray(xaxis)  # Setter not used - need asarray

        # Check assigned lenghts
        self.length_check()
        self.calibrated = calibrated
        self.header = header   # Access header with a dictionary call.

    @property
    def xaxis(self):
        # print("Getting xaxis property")
        return self._xaxis

    @xaxis.setter
    def xaxis(self, value):
        # print("xaxis value = ", value)
        if isinstance(value, str):
            # Try to catch some bad assignments
            # Yes a list of strings will not be caught
            raise TypeError("Cannot assign {} to the xaxis attribute".format(
                type(value)))
        elif value is None:
            try:
                # Try to assign arange the length of flux
                self._xaxis = np.arange(len(self._flux))
            except TypeError:
                # if self._flux is None then it has no length.
                self._xaxis = None
            # print("assigning xaxis the same length of _flux")

        # Add any other checks in here if necessary
        elif self._flux is not None:
            if len(value) != len(self._flux):
                raise ValueError("Lenght of xaxis does not match flux length")
            else:
                self._xaxis = np.asarray(value)
        else:
            # If flux is None
            self._xaxis = value

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, value):
        if isinstance(value, str):
            # Try to catch some bad assignments
            # Yes a list of strings will not be caught
            raise TypeError("Cannot assign {} to the flux attribute".format(
                type(value)))

        if value is not None:
            # print("Turning flux input into np array")
            # Not checking to make sure it equals the xaxis
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

            Note: This maight be better suited to return the new spectra
            instead of direct replacement.
        """
        x_org = self.xaxis
        flux_org = self.flux
        try:
            if len(self.xaxis) == 0:
                print("Warning! Spectrum has an empty xaxis to select"
                      " wavelengths from")
            else:
                mask = (self.xaxis > wav_min) & (self.xaxis < wav_max)
                self.flux = self.flux[mask]    # change flux first
                self.xaxis = self.xaxis[mask]
        except TypeError:
            print("Spectrum has no xaxis to select wavelength from")
            # Return to original values iscase were changed
            self.flux = flux_org           # Fix flux first
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
            print("Attribute xaxis is not wavelength calibrated."
                  " Cannot perform doppler shift")

    def calibrate_with(self, wl_map):
        """ Calibrate with polynomial with parameters wl_map.
        Input:
            wl_map - Polynomial cooeficients of form expected by np.poylval()
        Output:
            self.xaxis is replaced with the calibrated spectrum
            self.calibrated is set to True
        The parameters can be generated by np.polyfit(x, y, order)
        """
        if self.calibrated:
            # To bypass this could set Spectrum.calibrated = False
            raise SpectrumError("Spectrum is already calibrated"
                                ", Not recalibrating.")
        else:
            wavelength = np.polyval(wl_map, self.xaxis)   # Polynomial params
            if np.any(wavelength <= 0):
                raise SpectrumError("Wavelength solution contains zero or "
                                    "negative values. But wavelength must "
                                    "be positive. This solution will not "
                                    "doppler shift correctly. Please check "
                                    "your calibrations")
            else:
                self.xaxis = wavelength
                self.calibrated = True  # Set calibrated Flag

    def interpolate1d_to(self, reference, kind="linear", bounds_error=False,
                       fill_value=np.nan):
        """Interpolate wavelength solution to the  reference wavelength.
        Using scipy interpolation so the optional parameters are passed to
        scipy.
        See scipy.interolate.interp1d for more details
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d

        Inputs:
        reference : Spectrum or numpy.ndarray
        The reference to interpolate the xaxis to. Can either be a
        Spectrum type or a numpy.ndarray

        kind : str or int, optional
        Specifies the kind of interpolation as a string (‘linear’, ‘nearest’,
        ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’ where ‘slinear’, ‘quadratic’
        and ‘cubic’ refer to a spline interpolation of first, second or
        third order) or as an integer specifying the order of the spline
        interpolator to use. Default is ‘spline’.

        bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted
        on a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned fill_value.
        By default, an error is raised unless fill_value=”extrapolate”.

        fill_value : array-like or (array-like, array_like) or “extrapolate”,
        optional  (default : NaN)
        if a ndarray (or float), this value will be used to fill in for
        requested points outside of the data range. If not provided, then the
        default is NaN. The array-like must broadcast properly to the
        dimensions of the non-interpolation axes.
        If a two-element tuple, then the first element is used as a fill value
        for x_new < x[0] and the second element is used for x_new > x[-1].
        Anything that is not a 2-element tuple (e.g., list or ndarray,
        regardless of shape) is taken to be a single array-like argument meant
        to be used for both bounds as below, above = fill_value, fill_value.
        If “extrapolate”, then points outside the data range will be
        extrapolated. (“nearest” and “linear” kinds only.)

        """
        if kind == 'cubic':
            print("Warning! Cubic spline interpolation with interp1d can cause"
                  " memory errors and crashes")
        # Create scipy interpolation function from self
        interp_function = interp1d(self.xaxis, self.flux, kind=kind,
                                   fill_value=fill_value,
                                   bounds_error=bounds_error)

        # Determine the flux at the new locations given by reference
        if isinstance(reference, Spectrum):      # Spectrum type
            new_flux = interp_function(reference.xaxis)
            self.flux = new_flux                 # Flux needs to change first
            self.xaxis = reference.xaxis
        elif isinstance(reference, np.ndarray):  # Numpy type
            new_flux = interp_function(reference)
            self.flux = new_flux                 # Flux needs to change first
            self.xaxis = reference
        else:
            # print("Interpolate was not give a valid type")
            raise TypeError("Cannot interpolate with the given object of type"
                            " {}".format(type(reference)))

    # ###################R##################################
    # Overloading Operators
    # ######################################################

    def __add__(self, other):
        """ Overloaded addition method for Spectrum

        If there is addition between two Spectrum objects which have
        difference xaxis values then the second Spectrum is interpolated
        to the xaxis of the first Spectum

        e.g. if len(a.xaxis) = 10 and len(b.xaxis = 15)
        then if len(a + b) = 10 and len(b + a) = 15.

        This makes a + b != b + a

        """
        # Checks for type errors and size. It interpolates other if needed.
        prepared_other = self._prepare_other(other)
        new_flux = self.flux + prepared_other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __radd__(self, other):
        # E.g. for first Item in Sum  0  + Spectrum fails.

        new_flux = self.flux + other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __sub__(self, other):
        """ Overloaded subtraction method for Spectrum

        If there is subtraction between two Spectrum objects which have
        difference xaxis values then the second Spectrum is interpolated
        to the xaxis of the first Spectum

        e.g. if len(a.xaxis) = 10 and len(b.xaxis = 15)
        then if len(a - b) = 10 and len(b - a) = 15.

        # This makes a - b != -b + a

        """
        # Checks for type errors and size. It interpolates other if needed.
        prepared_other = self._prepare_other(other)
        new_flux = self.flux - prepared_other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __mul__(self, other):
        """ Overloaded multiplication method for Spectrum

        If there is multiplication between two Spectrum objects which have
        difference xaxis values then the second Spectrum is interpolated
        to the xaxis of the first Spectum

        e.g. if len(a.xaxis) = 10 and len(b.xaxis = 15)
        then if len(a * b) = 10 and len(b * a) = 15.

        This makes a * b != b * a

        """
        # Checks for type errors and size. It interpolates other if needed.
        prepared_other = self._prepare_other(other)
        new_flux = self.flux * prepared_other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __truediv__(self, other):
        """ Overloaded truedivision (/) method for Spectrum

        If there is truedivision between two Spectrum objects which have
        difference xaxis values then the second Spectrum is interpolated
        to the xaxis of the first Spectum

        e.g. if len(a.xaxis) = 10 and len(b.xaxis = 15)
        then if len(a / b) = 10 and len(b / a) = 15.

        This makes (a / b) != (1/b) / (1/a)

        """
        # Checks for type errors and size. It interpolates other if needed.
        prepared_other = self._prepare_other(other)
        # Divide by zero only gives a runtime warning with numpy
        new_flux = self.flux / prepared_other
        # May want to change the inf to something else, nan, 0?...
        # new_flux[new_flux == np.inf] = np.nan
        return Spectrum(flux=new_flux, xaxis=self.xaxis,
                        calibrated=self.calibrated)

    def __pow__(self, other):
        # Overlaod to use power to scale the flux of the spectra
        # if len(other) > 1 :
        #    raise ValueError("Spectrum can only be raised to the power of
        # one number not {}".format(len(other)))
        if isinstance(other, Spectrum):
            raise TypeError("Can not preform Spectrum ** Spectrum")
        elif isinstance(other, (int, float, np.ndarray)):
            try:
                new_flux = self.flux ** other
                return Spectrum(flux=new_flux, xaxis=self.xaxis,
                                header=self.header, calibrated=self.calibrated)
            except:
                # Type error or value error are likely
                raise
        else:
            raise TypeError("Unexpected type {} given for"
                            " __pow__".format(type(other)))

    def __len__(self):
        """ Return length of flux Spectrum"""
        return len(self.flux)

    def __neg__(self):
        """ Take negative flux """
        negflux = -self.flux
        return Spectrum(flux=negflux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __pos__(self):
        """ Take positive flux """
        posflux = +self.flux
        return Spectrum(flux=posflux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __abs__(self):
        """ Take absolute flux """
        absflux = abs(self.flux)
        return Spectrum(flux=absflux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def _prepare_other(self, other):
        if isinstance(other, Spectrum):
            if self.calibrated != other.calibrated:
                """Checking the Spectra are of same calibration state"""
                raise SpectrumError("Spectra are not calibrated similarly.")
            if np.all(self.xaxis == other.xaxis):  # Only for equal xaxis
                # Easiest condition in which xaxis of both are the same
                return copy.copy(other.flux)
            else:  # Uneven length xaxis need to be interpolated
                if ((np.min(self.xaxis) > np.max(other.xaxis)) |
                     (np.max(self.xaxis) < np.min(other.xaxis))):
                    raise ValueError("The xaxis do not overlap so cannot"
                                     " be interpolated")
                else:
                    other_copy = copy.copy(other)
                    #other_copy.interpolate_to(self)
                    return other_copy.flux
        elif isinstance(other, (int, float, np.ndarray)):
            return copy.copy(other)
        else:
            raise TypeError("Unexpected type {} given".format(type(other)))

# TO DO !
# --------------------
# Add an interpolation
# Incorporate interpolation into all overloaded operators

# Setter /getter for xaxis and flux to turn into np.asarrays


class SpectrumError(Exception):
    pass
