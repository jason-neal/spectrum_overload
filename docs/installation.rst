
.. _install:

=============================================
Installation
=============================================
Spectrum_overload is currently available from ``github``.

Navigate to where you want to put files.
Then Download:

with ``git``::

    $ git clone https://github.com/jason-neal/spectrum_overload.git

To install::

    $ cd spectrum_overload
    $ python setup.py install

The plan is to have it available via ``pip`` someday in the future.::

    # Bug me about this.
    $ pip install spectrum-overload

That day is not today...

If you have any issues with installation of spectrum_overload please open an `issue`_ so it can be resolved.

.. _issue:  https://github.com/jason-neal/spectrum_overload/issues


Requirements
============
The main requirements are

    - numpy
    - astropy
    - scipy
    - pyastronomy

If you are needing to use this package you probably have these installed already...

Other requirements are needed for running the `pytest <https://docs.pytest.org/en/latest/>`_ test suite.
If you want to try run the tests, run::

python setup.py test

after normal installation.

These other dependencies that you may need are

    - pytest-runner
    - hypothesis
    - pytest
    - pytest-cov
    - python-coveralls
    - coverage


Editable Installation
=====================
If you want to modify spectrum_overload you can instead install it like::

    $ python setup.py develop

..  or  pip install -e . when available
