.. Spectrum documentation master file, created by
   sphinx-quickstart on Sun Sep 11 23:45:23 2016.

.. image:: https://readthedocs.org/projects/spectrum-overload/badge/?version=latest
    :target: http://spectrum-overload.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/jason-neal/spectrum_overload.svg?branch=master
    :target: https://travis-ci.org/jason-neal/spectrum_overload

.. image:: https://coveralls.io/repos/github/jason-neal/spectrum_overload/badge.svg?branch=master
    :target: https://coveralls.io/github/jason-neal/spectrum_overload?branch=master

.. image:: https://codeclimate.com/github/jason-neal/spectrum_overload/badges/gpa.svg
    :target: https://codeclimate.com/github/jason-neal/spectrum_overload
    :alt: Code Climate

.. _home:

=============================================
Welcome to spectrum_overload's documentation!
=============================================

Spectrum_overload is to manipulate astronomical spectra in a spectrum class with :ref:`overloaded operators <overloaded_operators>`.
This means that you can easily divide and subtract spectra from each other using the math operators ``+``, ``-``, ``*``, ``/``, ``**`` keeping the wavelength, flux and headers together.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   classes


Other Projects
===============
There are many other packages that deal with spectra, none that I know of overload the operators.

In alphabetical order:

    - `astropy/specutils <https://github.com/astropy/specutils>`_
    - `cokelaer/spectrum <https://github.com/cokelaer/spectrum>`_
    - `crawfordsm/specreduce <https://github.com/crawfordsm/specreduce>`_
    - `pyspeckit/spectrum <https://github.com/pyspeckit/pyspeckit/tree/master/pyspeckit/spectrum>`_
    - `PySpectrograph/Spectra <https://github.com/crawfordsm/pyspectrograph/tree/master/PySpectrograph/Spectra>`_

I am sure there are others.


Contributions
=============
Any contributions and/or suggestions to improve this module are very welcome.

Submit `issues <https://github.com/jason-neal/spectrum_overload/issues>`_, suggestions or pull requests to `jason-neal/spectrum_overload <https://github.com/jason-neal/spectrum_overload>`_.

This is my first attempt at creating classes, and packaging a python project so any *helpful* feedback is appreciated.

Badges
======
**master**

.. image:: https://readthedocs.org/projects/spectrum-overload/badge/?version=latest
    :target: http://spectrum-overload.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://travis-ci.org/jason-neal/spectrum_overload.svg?branch=master
    :target: https://travis-ci.org/jason-neal/spectrum_overload
    :alt: Build Status
.. image:: https://coveralls.io/repos/github/jason-neal/spectrum_overload/badge.svg?branch=master
    :target: https://coveralls.io/github/jason-neal/spectrum_overload?branch=master
    :alt: Test Coverage
.. image:: https://codeclimate.com/github/jason-neal/spectrum_overload/badges/gpa.svg
    :target: https://codeclimate.com/github/jason-neal/spectrum_overload
    :alt: Code Climate
.. image:: https://codeclimate.com/github/jason-neal/spectrum_overload/badges/issue_count.svg
    :target: https://codeclimate.com/github/jason-neal/spectrum_overload
    :alt: Issue Count
.. image:: https://www.quantifiedcode.com/api/v1/project/6e918445f6f344c1af9c32f50718082e/badge.svg?branch=master
    :target: https://www.quantifiedcode.com/app/project/6e918445f6f344c1af9c32f50718082e
    :alt: Code Issues

**develop**

.. image:: https://readthedocs.org/projects/spectrum-overload/badge/?version=develop
    :target: http://spectrum-overload.readthedocs.io/en/develop
    :alt: Documentation Status
.. image:: https://travis-ci.org/jason-neal/spectrum_overload.svg?branch=develop
    :target: https://travis-ci.org/jason-neal/spectrum_overload?branch=develop
    :alt: Build Status
.. image:: https://coveralls.io/repos/github/jason-neal/spectrum_overload/badge.svg?branch=develop
    :target: https://coveralls.io/github/jason-neal/spectrum_overload?branch=develop
    :alt: Test Coverage
.. image:: https://codeclimate.com/github/jason-neal/spectrum_overload/badges/gpa.svg?branch=develop
    :target: https://codeclimate.com/github/jason-neal/spectrum_overload?branch=develop
    :alt: Code Climate GPA
.. image:: https://codeclimate.com/github/jason-neal/spectrum_overload/badges/issue_count.svg?branch=develop
    :target: https://codeclimate.com/github/jason-neal/spectrum_overload?branch=develop
    :alt: Issue Count
.. image:: https://www.quantifiedcode.com/api/v1/project/6e918445f6f344c1af9c32f50718082e/badge.svg?branch=develop
    :target: https://www.quantifiedcode.com/app/project/6e918445f6f344c1af9c32f50718082e?branch=develop
    :alt: Code Issues


.. note::
    To build the documentation locally go to the root `spectrum_overload` directory then run::

        pip install -e .[docs]   # editable installation, install docs dependencies.
        cd docs
        make html

    The documentation pages will be docs/.build/index.html.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
