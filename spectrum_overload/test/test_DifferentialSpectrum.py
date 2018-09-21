# -*- coding: utf-8 -*-

"""Test DifferentialSpectrum Class."""
from __future__ import division, print_function

import numpy as np

from spectrum_overload import DifferentialSpectrum, Spectrum


def test_assignment_of_differential():
    """Test that differential contains two spectra."""
    x = np.arange(2100, 2105, 0.5)
    y = np.random.random(len(x))
    spec_1 = Spectrum(xaxis=x, flux=y, calibrated=True)
    spec_2 = Spectrum(xaxis=x, flux=2 * y, calibrated=True)

    spec_diff = DifferentialSpectrum(spec_1, spec_2)

    assert spec_diff.spec1 == spec_1
    assert spec_diff.spec2 == spec_2


# TODO:
# Define a fixture that creates a differential with two spectra.
