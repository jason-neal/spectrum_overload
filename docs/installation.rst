
.. _install:

=============================================
Installation
=============================================
Spectrum_overload is currently available from ``github``.

Navigate to where you want to put files.
Then Download:

with ``git``::

    $ git clone https://github.com/jason-neal/spectrum_overload.git

To install, run::

    $ cd spectrum_overload
    $ python setup.py install

    # Or:

    $ cd spectrum_overload
    $ pip install .

    # Or, for an editable installation:

    $ cd spectrum_overload
    $ pip install -e .

The plan is to have it available via ``pypi`` someday in the future.::

    # Bug me about this.
    $ pip install spectrum-overload

That day is not today...

If you have any issues with installation of spectrum_overload please open an `issue`_ so it can be resolved.

.. _issue:  https://github.com/jason-neal/spectrum_overload/issues


Requirements
============
The main requirements are

    - `numpy <https://www.numpy.org/>`_
    - `astropy <https://www.astropy.org/>`_
    - `scipy <https://www.scipy.org/>`_
    - `pyastronomy <http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/index.html>`_

If you are needing to use this package you probably have these installed already...

Unfortunately `pyastronomy <http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/index.html>`_ cannot be added to the ``requirements.txt`` alongside ``numpy`` due to `issue #22 <https://github.com/sczesla/PyAstronomy/issues/22>`_ with the setup dependency of ``numpy`` in ``pyastronomy``.
If you do not have it installed already then run::

    $ pip install pyastronomy

Other requirements are needed for running the `pytest <https://docs.pytest.org/en/latest/>`_ test suite.
If you want to try run the tests, run::

    $ python setup.py test

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
