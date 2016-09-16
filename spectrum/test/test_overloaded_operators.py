#!/usr/bin/env python

from __future__ import division, print_function
import pytest
import numpy as np
from astropy.io import fits
from pkg_resources import resource_filename
import sys
# Add Spectrum location to path
#sys.path.append('../')
from spectrum import Spectrum

# Test using hypothesis
from hypothesis import given
import hypothesis.strategies as st



#######################################################
#    Overloading Operators
#######################################################


#Try just with integers
@given(st.lists(st.integers(min_value=-100000, max_value=100000), min_size=1),
 st.integers(min_value=-1000000, max_value=1000000), 
 st.integers(min_value=-1000000, max_value=1000000), st.booleans())
def test_overload_add_integers_with_same_xaxis(x1, y1, y2 ,calib):
    x1 = np.asarray(x1)
    y1 *= x1
    y2 *= x1

    spec1 = Spectrum.Spectrum(flux=y1, xaxis=x1, calibrated=calib)
    spec2 = Spectrum.Spectrum(flux=y2, xaxis=x1, calibrated=calib)
    
    spec3 = spec1+spec2
    spec4 = sum([spec1, spec2])
    spec5 = sum([spec1, spec2, spec3, spec4])
    summed = np.asarray(y1) + np.asarray(y2)
    npsummed = np.asarray(y1) + np.asarray(y2)
    # Assert the flux values are summed togther
    assert np.all(spec3.flux == summed)
    assert np.all(spec3.flux == spec4.flux)
    assert np.all(spec4.flux == summed)
    assert np.all(spec4.flux == npsummed)
    assert np.all(spec5.flux == 3*summed)
    
    # Assert calibration has stayed the same.
    assert np.allclose(spec4.calibrated, spec1.calibrated)
    assert np.allclose(spec4.calibrated, spec3.calibrated)
    assert np.allclose(spec3.calibrated, spec2.calibrated)

# Try with floats
@given(st.lists(st.floats(min_value=1e-3, allow_infinity=False), min_size=1,), st.floats(min_value=1e-3), st.floats(min_value=1e-3),st.booleans())
def test_overload_add_with_same_xaxis(x1, y1, y2 ,calib):
    x1 = np.asarray(x1)
    y1 *= x1
    y2 *= x1
    spec1 = Spectrum.Spectrum(flux=y1, xaxis=x1, calibrated=calib)
    spec2 = Spectrum.Spectrum(flux=y2, xaxis=x1, calibrated=calib)
    
    spec3 = spec1+spec2
    spec4 = sum([spec1, spec2])
    # Assert the flux values are summed togther
    assert np.allclose(spec3.flux, np.asarray(y1) + np.asarray(y2))
    assert np.allclose(spec3.flux, spec4.flux)
    
    # Assert calibration has stayed the same.
    assert np.allclose(spec4.calibrated, spec1.calibrated)
    assert np.allclose(spec4.calibrated, spec3.calibrated)
    assert np.allclose(spec3.calibrated, spec2.calibrated)
    # Need to also check on xaxis after the calibraion has been performed.




@given(st.lists(st.floats(min_value=1e-3, allow_infinity=False), min_size=1,), st.floats(min_value=1e-3), st.floats(min_value=1e-3),st.booleans())
def test_overload_add_with_same_xaxis(x1, y1, y2 ,calib):
    x1 = np.asarray(x1)
    y1 *= x1
    y2 *= x1
    spec1 = Spectrum.Spectrum(flux=y1, xaxis=x1, calibrated=calib)
    spec2 = Spectrum.Spectrum(flux=y2, xaxis=x1, calibrated=calib) 
    spec_add = spec1 + spec2

    # Assert the flux values are summed togther
    assert np.allclose(spec_add.flux, np.asarray(y1) + np.asarray(y2))
    
    #Testing some other random things between them
    assert np.all(spec_div.xaxis == spec_mul.xaxis)
    assert spec_div.calibrated == spec_sub.calibrated
    assert spec_add.calibrated == spec_mul.calibrated
    assert spec_mul.calibrated == spec2.calibrated
    assert spec_div.calibrated == spec1.calibrated
    assert spec_div.header == spec_sub.header     # Might not want this later. May want to record the transformation in the header


@given(st.lists(st.floats(min_value=1e-3, allow_infinity=False), min_size=1,), st.floats(min_value=1e-3), st.floats(min_value=1e-3),st.booleans())
def test_overload_sub_with_same_xaxis(x1, y1, y2 ,calib):
    x1 = np.asarray(x1)
    y1 *= x1
    y2 *= x1
    spec1 = Spectrum.Spectrum(flux=y1, xaxis=x1, calibrated=calib)
    spec2 = Spectrum.Spectrum(flux=y2, xaxis=x1, calibrated=calib) 
    spec_sub = spec1 - spec2
    
    assert np.allclose(spec_sub.flux, np.asarray(y1) - np.asarray(y2))
    
    #Testing some other random things between them
    assert np.all(spec_sub.xaxis == spec2.xaxis)
    assert np.all(spec_sub.xaxis == spec1.xaxis)
    assert spec_sub.calibrated == spec1.calibrated 
    assert spec_sub.calibrated == spec2.calibrated 
    assert spec_sub.header == spec1.header     # Might not want this later. May want to record the transformation in the header
    
@given(st.lists(st.floats(min_value=1e-3, allow_infinity=False), min_size=1,), st.floats(min_value=1e-3), st.floats(min_value=1e-3),st.booleans())
def test_overload_mul_with_same_xaxis(x1, y1, y2 ,calib):
    x1 = np.asarray(x1)
    y1 *= x1
    y2 *= x1
    spec1 = Spectrum.Spectrum(flux=y1, xaxis=x1, calibrated=calib)
    spec2 = Spectrum.Spectrum(flux=y2, xaxis=x1, calibrated=calib) 
    spec_mul = spec1 * spec2
    
    assert np.allclose(spec_mul.flux, np.asarray(y1) * np.asarray(y2))
    
    #Testing some other random things between them
    assert np.all(spec_mul.xaxis == spec2.xaxis)
    assert np.all(spec_mul.xaxis == spec1.xaxis)
    assert spec_mul.calibrated == spec1.calibrated
    assert spec_mul.calibrated == spec2.calibrated
    assert spec_mul.header == spec1.header     # Might not want this later. May want to record the transformation in the header
    

@given(st.lists(st.floats(min_value=1e-3, allow_infinity=False), min_size=1,), st.floats(min_value=1e-3), st.floats(min_value=1e-3),st.booleans())
def test_overload_truediv_with_same_xaxis(x1, y1, y2 ,calib):
    x1 = np.asarray(x1)
    y1 *= x1
    y2 *= x1
    spec1 = Spectrum.Spectrum(flux=y1, xaxis=x1, calibrated=calib)
    spec2 = Spectrum.Spectrum(flux=y2, xaxis=x1, calibrated=calib) 
    spec_truediv = spec1 / spec2
    
    assert np.allclose(spec_truediv.flux, np.asarray(y1) / np.asarray(y2))
    
    #Testing some other random things between them
    assert np.all(spec_truediv.xaxis == spec2.xaxis)
    assert np.all(spec_truediv.xaxis == spec1.xaxis)
    assert spec_truediv.calibrated == spec1.calibrated
    assert spec_truediv.calibrated == spec2.calibrated
    assert spec_truediv.header == spec1.header     # Might not want this later. May want to record the transformation in the header
    





    
def test_len_works():
    #Test len works
    spec1 = Spectrum.Spectrum([1,2,3,4,5],[1,2,3,4,5])
    assert len(spec1) == 5

#def test_for_raise_die_to_calibration_mismatch():
    #Try catch my raise
    # s1 = Spectrum.Spectrum([1], [2], calibrated=True)
    # s2 = Spectrum.Spectrum([1], [2], calibrated=False)
    # # This will fail untill I work out errors more
    # with pytest.raises(SpectrumCalibrationError):
    #     s1 + s2
    # with pytest.raises(SpectrumCalibrationError):
    #     s1 - s2
    # with pytest.raises(SpectrumCalibrationError):
    #     s1 * s2
    # with pytest.raises(SpectrumCalibrationError):
    #     s1 / s2


def test_overload_pow():
    # Trying to catch error with raises 
    z = 2
    spec1 = Spectrum.Spectrum([1,2,3,4], [2,3,4,5], None, True)
    spec2 = Spectrum.Spectrum([1,2,3,4], [1,3,1,4], None, True)
    
    with pytest.raises(ValueError):
        spec3 = spec1 ** spec2
    with pytest.raises(ValueError):
        spec3 = spec1 ** [1,2] # This should fail
    with pytest.raises(ValueError):
        spec3 = spec1 ** np.array([1,2])
    
    # Including the modulo value 
    assert np.all(pow(spec1, 5, 3) == ((spec1 ** 5)  % 3))