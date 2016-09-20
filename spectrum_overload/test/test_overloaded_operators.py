#!/usr/bin/env python

from __future__ import division, print_function
import pytest
import numpy as np
#from astropy.io import fits
#from pkg_resources import resource_filename
#import sys
# Add Spectrum location to path
#sys.path.append('../')
from spectrum_overload import Spectrum
from spectrum_overload.Spectrum import SpectrumError

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
def test_overload_add_integers_with_same_xaxis(x1, y1, y2, calib):
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
@given(st.lists(st.floats(min_value=1e-3, max_value=1e7, allow_infinity=False), min_size=1,), st.floats(min_value=1e-3), st.floats(min_value=1e-3), st.booleans())
def test_overload_add_with_same_xaxis(x1, y1, y2, calib):
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



@given(st.lists(st.floats(min_value=1e-3, max_value=1e7, allow_infinity=False), min_size=1,), st.floats(min_value=-1e7, max_value=1e7), st.floats(min_value=-1e10, max_value=1e10), st.booleans())
def test_overload_sub_with_same_xaxis(x1, y1, y2, calib):
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
    
@given(st.lists(st.floats(min_value=1e-3, max_value=1e7, allow_infinity=False), min_size=1,), st.floats(min_value=-1e7, max_value=1e7), st.floats(min_value=-1e10, max_value=1e10), st.booleans())
def test_overload_mul_with_same_xaxis(x1, y1, y2, calib):
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
    

@given(st.lists(st.floats(min_value=1e-3, max_value=1e7, allow_infinity=False), min_size=1,), st.floats(min_value=1e-3, max_value=1e7, allow_infinity=False), st.floats(min_value=-1e10, max_value=1e10, allow_infinity=False), st.booleans())
def test_overload_truediv_with_same_xaxis(x1, y1, y2, calib):
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
    
def test_truediv_with_number():
    # To test if can divide flux by a number
    number = 0.3
    flux_arr = np.array([1, 2, 3, 2.3, 4.5])
    spec1 = Spectrum.Spectrum(flux=flux_arr, xaxis=[1, 1.1, 1.2, 2.1, 4], calibrated=True)
    
    spec_truediv = spec1 / number
    
    assert np.all(spec_truediv.flux == flux_arr/number)



def test_len_works():
    #Test len works
    spec1 = Spectrum.Spectrum([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert len(spec1) == 5



def test_for_raise_die_to_calibration_mismatch():
    #Try catch my raise
    s1 = Spectrum.Spectrum([1], [2], calibrated=True)
    s2 = Spectrum.Spectrum([1], [2], calibrated=False)
     # This will fail untill I work out errors more
    with pytest.raises(SpectrumError):
         s1 + s2
    with pytest.raises(SpectrumError):
         s1 - s2
    with pytest.raises(SpectrumError):
         s1 * s2
    with pytest.raises(SpectrumError):
         s1 / s2


def test_overload_pow():
    # Trying to catch error with raises 
    power = 2
    spec1 = Spectrum.Spectrum([1, 2, 3, 4], [2, 3, 4, 5], None, True)
    spec2 = Spectrum.Spectrum([1, 2, 3, 4], [1, 3, 1, 4], None, True)
    # Can test when things are not suposed to work :)
    with pytest.raises(TypeError):
        spec1 ** spec2
    with pytest.raises(ValueError):
        spec1 ** [1, 2]  # This should fail
    with pytest.raises(ValueError):
        spec1 ** np.array([1, 2])
    # Should also test that something works
    spec4 = spec1 ** power
    assert np.all(spec4.flux == np.array([1, 4, 9, 16])) # flux is squared
    assert np.all(spec4.xaxis == spec1.xaxis)  # xaxis stays the same


@given(st.lists(st.floats(min_value=1e-3, max_value=1e7, allow_infinity=False), min_size=1,), st.floats(min_value=1e-3, max_value=1e7, allow_infinity=False), st.floats(min_value=1e-7, max_value=1e10, allow_infinity=False), st.integers(min_value=1, max_value=int(1e5)))
def test_add_sub_mult_divide_by_numbers(x, y, float1, int1):
    y *= np.array(x)   # turn to array for operations
    spec = Spectrum.Spectrum(flux=y, xaxis=x)
    # Add by a float 
    spec_add = spec + float1
    spec_add_int = spec + int1
    assert np.all(spec_add.flux == y + float1)
    assert np.all(spec_add_int.flux == y + int1)
    # Subtract by an int
    spec_sub = spec - float1
    spec_sub_int = spec - int1
    assert np.all(spec_sub.flux == y - float1)
    assert np.all(spec_sub_int.flux == y - int1)
    # Multiply by an int
    spec_mul = spec * float1
    spec_mul_int = spec * int1
    assert np.all(spec_mul.flux == y * float1)
    assert np.all(spec_mul_int.flux == y * int1)
    # Divide by an int
    spec_truediv = spec / float1
    spec_truediv_int = spec / int1
    assert np.all(spec_truediv.flux == y / float1)
    assert np.all(spec_truediv_int.flux == y / int1)


def test_unitary_operators():
    """ Test __pos__ and __neg__ operators"""
    a = np.array([1, 2, -3, 4])
    b = np.array([1, 2, 3, 4])
    spec = Spectrum.Spectrum(a, b)
    spec1 = +spec
    assert np.all(spec1.flux == a)
    assert np.all(spec1.flux == spec.flux)
    spec2 = -spec
    assert np.all(spec2.flux == -a)
    assert np.all(spec2.flux == -spec1.flux)
    

def test_abs_operator():
    """ Test absolute value of flux"""
    spec = Spectrum.Spectrum([-1, 2, -3.2, 4], [2, 3, 4, 5])
    abs_spec = abs(spec)
    abs_spec2 = abs(abs_spec)
    assert np.all(abs_spec.flux == np.array([1, 2, 3.2, 4]))
    assert np.all(abs_spec.flux == abs_spec2.flux)


    