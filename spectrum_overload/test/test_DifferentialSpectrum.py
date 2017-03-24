"""Test DifferentialSpectrum Class."""

import pytest
import numpy as np

from spectrum_overload.Spectrum import Spectrum
from spectrum_overload.Differential import DifferentialSpectrum


def test_assignment_of_differential():
    """Test that differential contains two spectra."""
    x = np.arange(2100, 2105, 0.5)
    y = np.random.random(len(x))
    spec_1 = Spectrum(x, y, calibrated=True)
    spec_2 = Spectrum(x, 2 * y, calibrated=True)

    spec_diff = DifferentialSpectrum(spec_1, spec_2)

    assert spec_diff.spec1 == spec_1
    assert spec_diff.spec2 == spec_2

# TODO:
# Define a fixture that creates a differentail with two spectra.