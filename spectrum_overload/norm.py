# -*- coding: utf-8 -*-

import numpy as np
import logging

def get_continuum_points(wave, flux, nbins=50, ntop=20):
    """Get continuum points along a spectrum.

    This splits a spectrum into "nbins" number of bins and calculates
    the median wavelength and flux of the upper "ntop" number of flux
    values.
    """
    # Shorten array until can be evenly split up.
    remainder = len(flux) % nbins
    if remainder:
        # Non-zero remainder needs this slicing
        wave = wave[:-remainder]
        flux = flux[:-remainder]

    wave_shaped = wave.reshape((nbins, -1))
    flux_shaped = flux.reshape((nbins, -1))

    s = np.argsort(flux_shaped, axis=-1)[:, -ntop:]

    s_flux = np.array([ar1[s1] for ar1, s1 in zip(flux_shaped, s)])
    s_wave = np.array([ar1[s1] for ar1, s1 in zip(wave_shaped, s)])

    wave_points = np.nanmedian(s_wave, axis=-1)
    flux_points = np.nanmedian(s_flux, axis=-1)
    assert len(flux_points) == nbins

    return wave_points, flux_points


def continuum(wave, flux, method='scalar', degree=None, nbins=50, ntop=20):
    """Fit continuum of flux.

    Parameters
    ----------
    method: str ("scalar")
        The function type, valid functions are "scalar", "linear",
        "quadratic", "cubic", "poly", and "exponential".
        Default "scalar".
    degree: int, None
       Degree of polynomial when method="poly". Default = None.
    nbins: int
        Number of bins to separate the spectrum into.
    ntop: int
        Number of highest points in bin to take median of.
"""
    if method not in ("scalar", "linear", "quadratic", "cubic", "poly", "exponential"):
        raise ValueError("Incorrect method for polynomial fit.")

    if method != "poly" and degree is not None:
        logging.warning("The degree={0} is not used with method={1} in continuum fitting.".format(degree, method))

    if method == "poly" and degree is None:
        raise ValueError("No degree specified for continuum method 'poly'.")

    if np.any(np.isnan(wave)) or  np.any(np.isnan(flux)):
        raise ValueError("There are Nan values in spectrum. Please remove first.")

    org_wave = wave[:]

    # Get continuum value in chunked sections of spectrum.
    wave_points, flux_points = get_continuum_points(wave, flux, nbins=nbins, ntop=ntop)

    poly_degree = {"scalar": 0, "linear": 1, "quadratic": 2, "cubic": 3, "poly": degree}

    if method == "exponential":
        z = np.polyfit(wave_points, np.log(flux_points), deg=1, w=np.sqrt(flux_points))
        p = np.poly1d(z)
        continuum_fit = np.exp(p(org_wave))   # Un-log the y values.
    else:
        z = np.polyfit(wave_points, flux_points, deg=poly_degree[method])
        p = np.poly1d(z)
        continuum_fit = p(org_wave)

    return continuum_fit
