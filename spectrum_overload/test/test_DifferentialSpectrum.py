# -*- coding: utf-8 -*-

"""Test DifferentialSpectrum Class."""
from __future__ import division, print_function

import numpy as np
import pytest

from spectrum_overload import DifferentialSpectrum
from spectrum_overload import Spectrum



@pytest.mark.parametrize("key, value1, value2", [
    ("EXPTIME", 180, 200),
    ("OBJECT", "HD30501", "Hd47474"),
    ("HIERARCH ESO INS SLIT1 WID", 0.2, 0.4)])
def test_compatibility_checking(key, value1, value2):
    """Test that differential contains two spectra."""
    header = {"EXPTIME": 180, "HIERARCH ESO INS SLIT1 WID": 0.1, "OBJECT": "HD3000"}

    hdr1 = {}
    hdr2 = {}
    hdr1.update(header.copy())
    hdr1.update({key: value1})
    hdr2.update(header.copy())
    hdr2.update({key: value2})
    assert hdr1 != hdr2

    spec_1 = Spectrum(header=hdr1)
    spec_2 = Spectrum(header=hdr2)
    assert spec_2.header != spec_1.header

    compatible = DifferentialSpectrum.check_compatibility(spec_1, spec_2)
    assert not compatible  # Key parameters in headers are different


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
# Define a fixture that creates a differentail with two spectra.
