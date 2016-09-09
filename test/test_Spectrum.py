#!/usr/bin/env python

from __future__ import division, print_function
import pytest
import numpy as np

import sys
# Add Spectrum location to path
sys.path.append('../')
import Spectrum

# Test using hypothesis
from hypothesis import given
import hypothesis.strategies as st

@given(st.lists(st.floats()), st.lists(st.floats()), st.booleans())
def test_spectrum_assigns_hypothesis_data(y, x, z):
    """Test that data was assigned to the correct attributes"""
    spec = Spectrum.Spectrum(y, x, z)
    assert np.all(spec.flux == y)
    assert np.all(spec.xaxis == x)
    assert spec.calibrated == z

def test_spectrum_assigns_data():
    """Test a manual example"""
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 1, 0.9, 0.95, 1, 1]
    calib_val = 0

    spec = Spectrum.Spectrum(y, x, calibrated=calib_val)
    assert np.all(spec.flux == y)
    assert np.all(spec.xaxis == x)
    assert spec.calibrated == calib_val

@given(st.lists(st.floats()), st.booleans(), st.floats(), st.floats())
def test_wav_select(x, calib, wav_min, wav_max):
    """Test some properties of wavelength selection"""
    # Create specturm
    y = np.copy(x)
    spec = Spectrum.Spectrum(y, xaxis=x, calibrated=calib)
    # Select wavelength values
    spec.wav_select(wav_min, wav_max)

    # All values in selected spectrum should be less than the max and greater than the min value.
    if isinstance(spec.xaxis, list):
        assert all([xval >= wav_min for xval in spec.xaxis])
        assert all([xval <= wav_max for xval in spec.xaxis])
    else:
        assert all(spec.xaxis >= wav_min)
        assert all(spec.xaxis <= wav_max)

def test_wav_select_example():
    """Manual test of a wavelength selection"""
    # Create specturm
    y = 2*np.random.random(20)
    x = np.arange(20)
    calib = False
    spec = Spectrum.Spectrum(y, xaxis=x, calibrated=calib)
    # Select wavelength values

    spec.wav_select(5, 11)
    # All values in selected spectrum should be less than the max and greater than the min value.
    assert all(spec.xaxis >= 5)
    assert all(spec.xaxis <= 11)
    assert all(spec.xaxis == np.arange(6, 11))
    assert all(spec.flux == y[np.arange(6, 11)])


    ##Also need to test asignment!
    # spec2 = spec.wav_selector()
@given(st.lists(st.floats(min_value=1e-2, allow_infinity=False), min_size=1), st.floats(allow_infinity=False, allow_nan=False), st.booleans())
def test_doppler_shift_with_hypothesis(x, RV, calib):
    x = np.asarray(x)
    y = np.random.random(len(x))

    spec = Spectrum.Spectrum(y, x, calib)
    # Apply Doppler shift of RV km/s.
    spec.doppler_shift(RV)

    if not spec.calibrated:
        assert np.allclose(spec.xaxis, x)
    else:
        tolerance = 1e-6
        if abs(RV) < tolerance:
            assert np.allclose(spec.xaxis, x)
        elif RV < 0:
            assert np.all(spec.xaxis < x)
        elif RV > 0:
            assert np.all(spec.xaxis > x)


def test_x_calibration_works():
    """ Simple test to check that the calibration works """
    "Setup the code "
    x = [1,2,3,4,5,6,7,8,9,10]
    x = [float(x_i) for x_i in x]
    y = np.ones_like(x)
    spec = Spectrum.Spectrum(y, x, False)

    #Easy test
    params = np.polyfit([1,5,10], [3,15,30], 1)

    spec.calibrate_with(params)

    assert spec.calibrated
    assert np.allclose(spec.xaxis, np.asarray(x)*3)



