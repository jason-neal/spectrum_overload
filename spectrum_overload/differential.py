# -*- coding: utf-8 -*-

"""Differential Class which takes the difference between two spectra."""
from spectrum_overload import Spectrum


# Begin Feburary 2017
# Jason Neal

# TODO: Add in s-profile from
# Ferluga 1997: Separating the spectra of binary stars-I. A simple method: Secondary reconstruction

class DifferentialSpectrum(object):
    """A differential spectrum."""

    def check_compatibility(spec1, spec2):
        """Check spectra are compatible to take differences.

        Requires most of the setting to be the same. Have included a CRIRES only parameters also.
        """
        compatible = True
        for check in ["EXPTIME", "HIERARCH ESO INS SLIT1 WID", "OBJECT"]:
            if spec1.header[check] != spec1.header[check]:
                print("The Spectral property '{}' are not compatible. {}, {}".format(check, spec1.header[check],
                                                                                     spec2.header[check]))
                compatible = False
        return compatible

    def __init__(self, spectrum1, spectrum2, params=None):
        """Initialise lass with both spectra."""
        assert isinstance(spectrum1, Spectrum) and isinstance(spectrum2, Spectrum)
        if not (spectrum1.calibrated and spectrum2.calibrated):
            raise ValueError("Input spectra are not calibrated.")

        if self.check_compatibility(spectrum1, spectrum2):
            self.spec1 = spectrum1
            self.spec2 = spectrum2
            self.diff = None
        else:
            raise ValueError("The spectra are not compatible.")

    def barycentric_correct(self):
        """Barycentic correct each spectra."""
        pass

    def times(self):
        time = self.spec1.header["DATE-OBS"]
        time2 = self.spec2.header["DATE-OBS"]
        # TODO: Turn into datetime?
        return (time, time2)

    def rest_frame(self, frame):
        """Change restframe to one of the spectra."""
        pass

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
        return self.params

    @params.setter
    def params(self, value):
        """Params setter.

        A dictionary of orbital parameters to use for shifting frames.
        """
        if isinstance(value, dict):
            self.params = value
        else:
            raise TypeError("Orbital parameters need to be a dict.")
