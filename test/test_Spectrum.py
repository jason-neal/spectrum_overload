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

    spec = Spectrum.Spectrum(y, x, z)
    assert spec.flux == y
    assert spec.xaxis == x
    assert spec.calibrated == z

def test_spectrum_assigns_data():

    x = [1, 2, 3, 4, 5, 6]
    y = [1, 1, 0.9, 0.95, 1, 1]
    calib_val = 0

    spec = Spectrum.Spectrum(y, x, calibrated=calib_val)
    assert spec.flux == y
    assert spec.xaxis == x
    assert spec.calibrated == calib_val

@given(st.lists(st.floats()), st.lists(st.floats()), st.booleans(), st.floats(), st.floats())
def test_wav_select(y, x, calib, wav_min, wav_max):
    # Create specturm
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
