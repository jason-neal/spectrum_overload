"""Differential Class which takes the difference between two spectral types."""
from spectrum_overload.Spectrum import Spectrum
# Begin Feburary 2017


class DifferentialSpectrum(object):
    """Make differential spectrum."""

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
        """Acecpt dictionary of orbital parameters to use for shifting frames."""
        self.params = params
