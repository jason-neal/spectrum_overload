# -*- coding: utf-8 -*-

"""Differential Class which takes the difference between two spectra."""
import logging

import ephem
import matplotlib.pyplot as plt

from spectrum_overload import Spectrum


# Begin February 2017
# Jason Neal

# TODO: Add in s-profile from
# Ferluga 1997: Separating the spectra of binary stars-I. A simple method: Secondary reconstruction

class DifferentialSpectrum(object):
    """A differential spectrum."""

    @staticmethod
    def check_compatibility(spec1, spec2):
        """Check spectra are compatible to take differences.

        Requires most of the setting to be the same. Have included a CRIRES only parameters also.
        """
        for check in ["EXPTIME", "HIERARCH ESO INS SLIT1 WID", "OBJECT"]:
            if spec1.header[check] != spec2.header[check]:
                print("The Spectral property '{}' are not compatible. {}, {}".format(check, spec1.header[check],
                                                                                     spec2.header[check]))
                return False
        return True

    def __init__(self, spectrum1, spectrum2, params=None):
        """Initialise class with both spectra."""
        assert isinstance(spectrum1, Spectrum) and isinstance(spectrum2, Spectrum)
        if not (spectrum1.calibrated and spectrum2.calibrated):
            raise ValueError("Input spectra are not calibrated.")

        if self.check_compatibility(spectrum1, spectrum2):
            self.spec1 = spectrum1
            self.spec2 = spectrum2
            self._diff = self.diff
        else:
            raise ValueError("The spectra are not compatible.")
        self.params = params

    def barycentric_correct(self):
        """Barycentric correct each spectra.

         Based off CRIRES headers.
         """
        if not self.spec1.header.get("BERVDONE", False):
            self.spec1 = self.barycorr_spectrum(self.spec1)
            self.spec1.header.update({"BERVDONE": True})
            print("spec1 barycorrected")

        if not self.spec2.header.get("BERVDONE", False):
            self.spec2 = self.barycorr_spectrum(self.spec2)
            self.spec1.header.update({"BERVDONE": True})
            print("spec2 barycorrected")
        return None

    def times(self):
        time = self.spec1.header["DATE-OBS"]
        time2 = self.spec2.header["DATE-OBS"]
        # TODO: Turn into datetime?
        return (time, time2)

    def rest_frame(self, frame):
        """Change rest frame to one of the spectra."""
        pass

    @property
    def diff(self):
        """Calculate difference between the two spectra."""
        if self.check_compatibility(self.spec1, self.spec2):
            return self.spec1 - self.spec2
        else:
            raise ValueError("The spectra are not compatible.")
            # TODO: Access interpolations

    def sort(self, method="time"):
        """Sort spectra in specific order. e.g. time, reversed."""
        pass

    def swap(self):
        """Swap order of the two spectra."""
        self.spec1, self.spec2 = self.spec2, self.spec1

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        """Params setter.

        A dictionary of orbital parameters to use for shifting frames.
        """
        if isinstance(value, dict) or (value is None):
            self._params = value
        else:
            raise TypeError("Orbital parameters need to be a dict.")

    def plot(self, axis=None, **kwargs):
        """Plot spectrum with matplotlib."""
        if axis is None:
            plt.plot(self.diff.xaxis, self.diff.flux, **kwargs)
            if self.diff.calibrated:
                plt.xlabel("Wavelength")
            else:
                plt.xlabel("Pixels")
            plt.ylabel(r"\Delta Flux")
        else:
            axis.plot(self.diff.xaxis, self.diff.flux, **kwargs)

    @staticmethod
    def barycorr_spectrum(spectrum, extra_offset=None):
        """Wrapper to apply barycorrection of spectra if given a Spectrum object.

        Based off CRIRES headers."""
        _, nflux = self.barycorr_crires(spectrum.xaxis, spectrum.flux,
                                        spectrum.header, extra_offset=extra_offset)
        new_spectrum = Spectrum(flux=nflux, xaxis=spectrum.xaxis, header=spectrum.header)
        new_spectrum.header.update({"BERVDONE": True})
        return new_spectrum

    @staticmethod
    def barycorr_crires(wavelength, flux, header, extra_offset=None):
        """Calculate Heliocentric correction values and apply to spectrum.

        Based off CRIRES headers.
        """
        if header is None:
            logging.warning("No header information to calculate heliocentric correction.")
            header = {}
            if (extra_offset is None) or (extra_offset == 0):
                return wavelength, flux

        try:
            longitude = float(header["HIERARCH ESO TEL GEOLON"])
            latitude = float(header["HIERARCH ESO TEL GEOLAT"])
            altitude = float(header["HIERARCH ESO TEL GEOELEV"])

            ra = header["RA"]  # CRIRES RA already in degrees
            dec = header["DEC"]  # CRIRES hdr DEC already in degrees

            time = header["DATE-OBS"]  # Observing date  '2012-08-02T08:47:30.8425'

            # Convert easily to julian date with ephem
            jd = ephem.julian_date(time.replace("T", " ").split(".")[0])

            # Calculate Heliocentric velocity
            helcorr = pyasl.helcorr(longitude, latitude, altitude, ra, dec, jd,
                                    debug=False)
            helcorr = helcorr[0]
        except KeyError as e:
            logging.warning("Not a valid header so can't do automatic correction.")

            helcorr = 0.0

        if extra_offset is not None:
            logging.warning("Warning!!!! have included a manual offset for testing")
        else:
            extra_offset = 0.0

        helcorr_val = helcorr + extra_offset

        if helcorr_val == 0:
            logging.warning("Helcorr value was zero")
            return wavelength, flux
        else:
            # Apply Doppler shift to the target spectra with helcorr correction velocity
            nflux, wlprime = pyasl.dopplerShift(wavelength, flux, helcorr_val,
                                                edgeHandling=None, fillValue=None)

            logging.info("RV Size of Heliocenter correction for spectra", helcorr_val)
            return wlprime, nflux
