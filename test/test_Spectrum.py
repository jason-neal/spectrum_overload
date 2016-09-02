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

@given(st.lists(st.integers()), st.lists(st.floats()), st.lists(st.floats()))
def test_spectrum_assigns_hypothesis_data(x, y, z):

    spec = Spectrum.Spectrum(x, y, z)
    assert spec.flux == y
    assert spec.pixel == x
    assert spec.wavelength == z

def test_spectrum_assigns_data():

    x = [1,2,3,4,5,6]
    y = [1,1,0.9,0.95,1,1]
    z = 2200*x

    spec = Spectrum.Spectrum(x, y, z)
    assert spec.flux == y
    assert spec.pixel == x
    assert spec.wavelength == z
