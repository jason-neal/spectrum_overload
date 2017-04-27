#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licensed under the MIT Licence

# About spectrum_overload
# Based off of the warehouse project (pip's replacement)

import os.path

__all__ = [
    "__title__", "__summary__", "__uri__", "__version__", "__commit__",
    "__author__", "__email__", "__license__", "__copyright__",
]


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = None


__title__ = "spectrum_overload"
__summary__ = "A spectrum class that overloads operators"
# __uri__ = "https://pypi.org/"
__uri__ = None

# The version as used in the setup.py and the docs conf.py
__version__ = "0.1.0"

if base_dir is not None and os.path.exists(os.path.join(base_dir, ".commit")):
    with open(os.path.join(base_dir, ".commit")) as fp:
        __commit__ = fp.read().strip()
else:
    __commit__ = None

__author__ = "Jason Neal"
__email__ = "jason.neal@astro.up.pt"

__license__ = "MIT Licence"
__copyright__ = "2016 {0!s}".format(__author__)
