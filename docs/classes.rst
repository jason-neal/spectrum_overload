.. _classes

=================
Available Classes
=================
Currently there are two classes available.
    - :ref:`Spectrum <spectrumclass>`
    - :ref:`DifferentialSpectrum <diffclass>`


.. _spectrumclass:

Spectrum
=========

A class to contain a spectrum. The operators have been overloaded so that you can easily manipulate spectra.

.. autoclass:: spectrum_overload.spectrum.Spectrum
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
