""" spectrum_overload Setup.py
My first attempt at a setup.py file. It is based off

A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

"""
# Licensed under the MIT Licence

# To use a consistent encoding
import codecs
import os

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with codecs.open(os.path.join(here, "README.md")) as f:
    long_description = f.read()

about = {}
with codecs.open(os.path.join(here, "spectrum_overload", "__about__.py")) as f:
    exec(f.read(), about)

# https://www.reddit.com/r/Python/comments/3uzl2a/setuppy_requirementstxt_or_a_combination/
with codecs.open(os.path.join(here, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="spectrum_overload",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=about["__version__"],
    description="Spectrum class that overloads operators.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url="https://github.com/jason-neal/spectrum_overload",
    download_url="https://github.com/jason-neal/spectrum_overload",
    # Author details
    author=about["__author__"],
    author_email=about["__email__"],
    license=about["__license__"],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Natural Language :: English",
    ],
    keywords=["astronomy", "spectra", "spectroscopy", "CRIRES"],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    # test_suite=[],
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    # py_modules=["spectrum/Spectrum"],
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # install_requires=requirements,
    install_requires=["numpy", "scipy", "astropy", "pyastronomy", "matplotlib"],
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4",
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "hypothesis", "pytest-cov"],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage", "pytest", "pytest-cov", "python-coveralls", "hypothesis"],
        "docs": ["sphinx >= 1.4", "sphinx_rtd_theme", "pyastronomy"],
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={"spectrum_overload": ["data/*.fits"]},
    #    'sample': ['package_data.dat'],
    # },
    include_package_data=True,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        #    'console_scripts': [
        #        'sample=sample:main',
        "console_scripts": ["spectrum_overload=spectrum_overload:main"]
    },
)
