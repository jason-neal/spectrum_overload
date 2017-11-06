# -*- coding: utf-8 -*-

"""Differential Class which takes the difference between two spectra."""
from spectrum_overload import Spectrum

# Begin Feburary 2017
# Jason Neal

# TODO: Add in s-profile from
# Ferluga 1997: Separating the spectra of binary stars-I. A simple method: Secondary reconstruction

class DifferentialSpectrum(object):
    """A differential spectrum."""

    def __init__(self, Spectrum1, Spectrum2):
        """Initalise lass with both spectra."""
        if not(Spectrum1.calibrated and Spectrum2.calibrated):
            raise ValueError("Input spectra are not calibrated.")

        self.spec1 = Spectrum1
        self.spec2 = Spectrum2
        self.params = None
        self.diff = None

    def barycentric_correct(self):
        """Barycentic correct each spectra."""
        pass

    def rest_frame(self, frame):
        """Change restframe to one of the spectra."""
        pass

    def diff(self):
        """Calculate difference between the two spectra."""
        # TODO: Access interpolations
        return self.spec1 - self.spec2

    def sort(self, method="time"):
        """Sort spectra in specific order. e.g. time, reversed."""
        pass

    def swap(self):
        """Swap order of the two spectra."""
        self.spec1, self.spec2 = self.spec2, self.spec1

    def add_orbital_params(self, params):
        """A dictionary of orbital parameters to use for shifting frames."""
        self.params = params
