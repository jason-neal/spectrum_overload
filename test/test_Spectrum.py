#!/usr/bin/env python

from __future__ import division, print_function
import pytest

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

    x = [1,2,3,4,5,6]
    y = [1,1,0.9,0.95,1,1]
    calib_val = 0

    spec = Spectrum.Spectrum(y, x, calibrated=calib_val)
    assert spec.flux == y
    assert spec.xaxis == x
    assert spec.calibrated == calib_val