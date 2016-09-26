#!/usr/bin/env python

from __future__ import division, print_function
import copy
import pytest
import numpy as np
from astropy.io import fits
from pkg_resources import resource_filename

# import sys
# Add Spectrum location to path
# sys.path.append('../')
from spectrum_overload.Spectrum import Spectrum
# from spectrum_overload.Spectrum import SpectrumError

# Test using hypothesis
from hypothesis import given
import hypothesis.strategies as st


@given(st.lists(st.floats(allow_infinity=False, allow_nan=False)),
       st.integers(), st.booleans())
def test_spectrum_assigns_hypothesis_data(y, x, z):
    """Test that data was assigned to the correct attributes"""
    # Use one hypotheseis list they need to have the same lenght
    # multiply by a random int to mix it up a little.
    x = x * np.array(y)
    spec = Spectrum(y, x, calibrated=z)
    assert np.all(spec.flux == y)
    assert np.all(spec.xaxis == x)
    assert spec.calibrated == z


def test_spectrum_assigns_data():
    """Test a manual example
    Lenghts of x and y need to be the same"""
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 1, 0.9, 0.95, 1, 1]
    calib_val = 0

    spec = Spectrum(y, x, calibrated=calib_val)
    assert np.all(spec.flux == y)
    assert np.all(spec.xaxis == x)
    assert spec.calibrated == calib_val


def test_empty_call_is_nones():
    # Check empty Spectrum is implmented with Nones
    # and not Nones in an array (!=None)
    s = Spectrum()
    assert s.flux is None
    assert s.xaxis is None
    assert s.header is None
    assert s.calibrated is False


def test_setters_for_flux_and_xaxis():
    s = Spectrum()
    # Try set flux to None
    s.flux = None
    # Try set xaxis when flux is None
    s.xaxis = [1, 2, 3, 4]

    s.flux = [2, 2.1, 2.2, 2.1]
    # Spectrum(False,False)
    # Spectrum(False,None)
    # Spectrum(False,False)
    # Spectrum(None,False)
    pass


def test_length_checking():
    s = Spectrum(flux=None, xaxis=None)
    # Length check when both None
    s.length_check()
    # Try set xaxis when flux is None
    s.xaxis = [1, 2, 3, 4]
    # Run length check when flux is None and xaxis is not
    s.length_check()
    s.flux = [2, 2.1, 2.2, 2.1]

    # Create new empty spectrum
    s = Spectrum(flux=None, xaxis=None)
    s.flux = [2, 2.1, 2.2, 2.1]
    # Run length check when xaxis is None and flux is not
    print(s.xaxis)
    print(s.flux)
    s.length_check()

    with pytest.raises(ValueError):
        # Wrong length should fail
        Spectrum([1, 4, 5], [2, 1])


def test_flux_and_xaxis_cannot_pass_stings():
    """Passing a string to flux or xaxis will raise a TypeError"""
    with pytest.raises(TypeError):
        Spectrum([1, 2, 3], xaxis='bar')
    with pytest.raises(TypeError):
        Spectrum("foo", [1.2, 3, 4, 5])
    with pytest.raises(TypeError):
        Spectrum("foo", "bar")
    spec = Spectrum([1, 1, .5, 1])
    with pytest.raises(TypeError):
        spec.flux = "foo"
    with pytest.raises(TypeError):
        spec.xaxis = 'bar'


def test_auto_genration_of_xaxis_if_None():
    spec = Spectrum([1, 1, .5, 1])
    assert np.all(spec.xaxis == np.arange(4))
    spec2 = Spectrum([1, 1, .5, 1], [100, 110, 160, 200])
    spec2.xaxis = None  # reset xaxis
    assert np.all(spec2.xaxis == np.arange(4))


def test_length_of_flux_and_xaxis_equal():
    """ Try assign a mismatched xaxis it should raise a ValueError"""
    with pytest.raises(ValueError):
        Spectrum([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        Spectrum([1, 2, 3], [])
    with pytest.raises(ValueError):
        Spectrum([], [1, 2])
    spec = Spectrum([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        spec.xaxis = [1, 2]


@given(st.lists(st.floats()), st.booleans(), st.floats(), st.floats())
def test_wav_select(x, calib, wav_min, wav_max):
    """Test some properties of wavelength selection"""
    # Create specturm
    y = np.copy(x)
    spec = Spectrum(y, xaxis=x, calibrated=calib)
    # Select wavelength values
    spec.wav_select(wav_min, wav_max)

    # All values in selected spectrum should be less than the max and greater
    # than the min value.
    assert all(spec.xaxis >= wav_min)
    assert all(spec.xaxis <= wav_max)


def test_wav_select_example():
    """Manual test of a wavelength selection"""
    # Create specturm
    y = 2*np.random.random(20)
    x = np.arange(20)
    calib = False
    spec = Spectrum(y, xaxis=x, calibrated=calib)
    # Select wavelength values

    spec.wav_select(5, 11)
    # All values in selected spectrum should be less than the max and greater
    # than the min value.
    assert all(spec.xaxis >= 5)
    assert all(spec.xaxis <= 11)
    assert all(spec.xaxis == np.arange(6, 11))
    assert all(spec.flux == y[np.arange(6, 11)])
    # Also need to test asignment!
    # spec2 = spec.wav_selector()


@given(st.lists(st.floats(min_value=1e-6, allow_infinity=False), min_size=1),
       st.floats(), st.booleans())
def test_doppler_shift_with_hypothesis(x, RV, calib):
    """Test doppler shift properties.
    Need to check values against pyastronomy separately """
    x = np.asarray(x)
    y = np.random.random(len(x))

    spec = Spectrum(y, x, calibrated=calib)
    # Apply Doppler shift of RV km/s.
    spec.doppler_shift(RV)

    if not spec.calibrated:
        assert np.allclose(spec.xaxis, x)
    else:
        tolerance = 1e-6
        if abs(RV) < tolerance:
            assert np.allclose(spec.xaxis, x)
        elif np.isnan(RV) or np.isinf(RV):
            assert np.all(spec.xaxis == x)
        elif RV < 0:
            assert np.all(spec.xaxis < x)
        elif RV > 0:
            assert np.all(spec.xaxis > x)


def test_x_calibration_works():
    """ Simple test to check that the calibration works """
    "Setup the code "
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = [float(x_i) for x_i in x]
    y = np.ones_like(x)
    spec = Spectrum(y, x, False)

    # Easy test
    params = np.polyfit([1, 5, 10], [3, 15, 30], 1)

    spec.calibrate_with(params)

    assert spec.calibrated
    assert np.allclose(spec.xaxis, np.asarray(x)*3)


def test_header_attribute():
    """Test header attribute is accessable as a dict"""
    header = {"Date": "20120601", "Exptime": 180}
    spec = Spectrum(header=header)
    # Some simple assignment tests
    assert spec.header["Exptime"] == 180
    assert spec.header["Date"] == "20120601"

    # Try with a Astropy header object
    test_file = resource_filename('spectrum_overload', 'data/spec_1.fits')
    fitshdr = fits.getheader(test_file)
    spec2 = Spectrum(header=fitshdr)

    assert spec2.header["OBJECT"] == fitshdr["OBJECT"]
    assert spec2.header["EXPTIME"] == fitshdr["EXPTIME"]

    assert Spectrum().header is None  # unassign header is None


def test_interpolation():
    # Test the interpolation function some how
    # simple examples?
    # simple linear case
    x1 = [1., 2., 3., 4., 5.]
    y1 = [2., 4., 6., 8., 10]
    x2 = [1.5, 2, 3.5, 4]
    y2 = [1., 2., 1., 2.]
    S1 = Spectrum(y1, x1)
    S2 = Spectrum(y2, x2)
    S_lin = copy.copy(S1)
    S_lin.interpolate_to(S2, kind='linear')

    assert np.allclose(S_lin.flux, [3., 4., 7., 8.])
    # test linear interpoation matches numpy interp
    assert np.allclose(S_lin.flux, np.interp(x2, x1, y1))

    S_same = copy.copy(S1)
    # Interpolation to itself should be the same
    S_same.interpolate_to(S1)
    assert np.allclose(S_same.flux, S1.flux)
    assert np.allclose(S_same.xaxis, S1.xaxis)

    # Need to test that if boundserror is True a ValueError is raised
    with pytest.raises(ValueError):
        S2.interpolate_to(S1, bounds_error=True)
    with pytest.raises(ValueError):
        S2.interpolate_to(S1, kind='linear', bounds_error=True)
    with pytest.raises(TypeError):
        S2.interpolate_to(x1, bounds_error=True)
    with pytest.raises(ValueError):
        S2.interpolate_to(np.asarray(x1), bounds_error=True)
    with pytest.raises(TypeError):
        S2.interpolate_to([1, 2, 3, 4])
    with pytest.raises(TypeError):
        S2.interpolate_to("string")
    # Need to write better tests!


# test_doppler_shift_with_hypothesis()
