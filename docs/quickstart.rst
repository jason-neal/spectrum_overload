.. _quickstart:

==================
Quickstart Guide
==================

First install spectrum_overload via the :ref:`installation guide <install>`.


Import the Spectrum class into your code.

::

    from spectrum_overload.Spectrum import Spectrum

Then to create a spectrum object, and exponentially scale it by 2 use:

::

    s = Spectrum(flux=f, xaxis=x)

    # exponential scaling
    s = s ** 2

where ``f`` and ``x`` are 1D lists or numpy arrays.

.. note::

	Flux is the first kwarg. Be careful not to switch the order of flux and xaxis when not specifying the kwarg names.

By default the spectrum is "calibrated". If the xaxis is the pixel values of your spectra you can pass

::

    s = Spectrum(flux=f, xaxis=pxls, calibrated=False)

or if no xaxis is given it will be set by

::

    xaxis = np.arange(len(flux))

You can calibrate the spectra with a polynomial using the ``calibrate_with`` method which uses ``np.polyval``::

    s.calibrate_with([p0, p1, p2 ...])


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
    If the spectra are do not have the same wavelength axis then it is automatically interpolated
    to match the first spectrum or to another defined new xaxis.
