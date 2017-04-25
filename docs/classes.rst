.. _classes

=================
Available Classes
=================
Currently there are two classes available.
    - :ref:`Spectrum <spectrumclass>`
    - :ref:`DifferentialSpectrum <diffclass>`


.. _spectrumclass:

.. note::
    Unfortunately the auto-documentation for these classes does not build properly yet due to the ``Pyastronomy`` install dependency `issue. <https://github.com/sczesla/PyAstronomy/issues/22>`_
    I hope to see it here soon.

    If you need you can try and build this documentation yourself from the root `spectrum_overload` directory::

        pip install -e .[docs]   # editable installation, install docs dependencies.
        cd docs
        make html

    This page will be docs/.build/classes.html

Spectrum
=========

A class to contain a spectrum. The operators have been overloaded so that you can easily manipulate spectra.

.. autoclass:: spectrum_overload.Spectrum.Spectrum
   :members:
   :undoc-members:
   :show-inheritance:


.. _diffclass:

Differential Spectrum
=====================
Compute the difference between two :ref:`Spectrum <spectrumclass>` objects.

This is in an introductory state and need more work.

Would be useful add an s-profile function from `Ferluga et al. 1997 <http://aas.aanda.org/articles/aas/ps/1997/01/dst6676.ps.gz>`_. The subtraction of two gaussian lines with a RV offset.

.. autoclass:: spectrum_overload.Differential.DifferentialSpectrum
   :members:
   :undoc-members:
   :show-inheritance:
