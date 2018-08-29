.. _quickstart:

==================
Quickstart Guide
==================
First install spectrum_overload via the :ref:`installation guide <install>`.

Import the Spectrum class into your code.

::

    from spectrum_overload import Spectrum

Then to create a spectrum object, and raise to the power of 2 use:

::

    s = Spectrum(flux=f, xaxis=x)

    # power law scaling
    s = s ** 2

where ``f`` and ``x`` are 1D lists or numpy arrays.

.. note::

	Flux is the first kwarg. Be careful not to switch the order of flux and xaxis when not specifying the kwarg names.

By default the spectrum is "calibrated". If the xaxis is only the pixel values of your spectra you can pass:

::

    s = Spectrum(flux=f, xaxis=pxls, calibrated=False)

or if no xaxis is given it will be set by

::

    xaxis = np.arange(len(flux))

You can calibrate the spectra with a polynomial using the ``calibrate_with`` method which uses ``np.polyval``::

    s.calibrate_with([p0, p1, p2 ...])

Some methods are only available when you have a wavelength calibrated spectra, such as Doppler shifting with ``Spectrum.doppler_shift()``.

.. _normalization:

Normalization
=============
You can continuum normalize the spectrum using the ``Spectrum.normalize()`` method. It does not overwrite the spectrum but returns a new spectrum.
Possible methods include ``scalar``, ``linear``, ``quadratic``, ``cubic``, ``poly``, ``exponential``. The ``poly`` method requires a ``degree`` to also be provided.

E.g.::

    s = Spectrum(...)
    s = s.normalize("linear")
    # or
    s = s.normalize("poly", degree=1)

The normalization happens by dividing th spectrum by the fitted continuum.


.. _continuum_fitting:

Continuum fitting
*****************
The fitting of the continuum is rather tricky due to the presence of absorption lines. The continuum is fitted by dividing the spectrum into N even bins.
The highest M points out of each bin are chosen to represent the continuum for that bin. Their median/mean wavelength and flux position are used for each bin.
A fit across the N bins is then performed using the specified method.
The arguments ``nbins`` and ``ntop`` are used to specify the N and M values and can be passed to the ``normalize`` and ``continuum`` methods.

::

    continuum = s.continuum(method"poly", degree=2, nbins=50, ntop=15)

You probably need to change these parameters to for the size/length of your spectrum.

.. note ::

    Its probably best to experiment with the best ``nbins`` and ``ntop`` parameters for your spectrum.


.. _overloaded_operators:

Overloaded Operators
====================
The main purpose of this package is overloading the operators ``+``, ``-``, ``*``, ``/``, ``**`` to work with spectrum objects. e.g::

    # Given two spectra
    s1 = Spectrum(...)
    s2 = Spectrum(...)

    # You can easily do the following operations.
    add = s1 + s2
    subtract = s1 - s2
    multiply = s1 * s2
    divide = s1 / s2
    power = s1 ** a  # where a is a number


This is to make easier to do some basic manipulation of spectra, average spectra, take the difference, normalization,
exponential scaling etc...

.. note ::

    Its probably best to interpolate the spectra to the same xaxis yourself before hand.
    If the spectra do not have the same wavelength axis then it is automatically spline interpolated
    to match the first spectrum or to another defined new xaxis.
