import argparse
import sys

import matplotlib.pyplot as plt
from astropy.io import fits

from spectrum_overload import Spectrum, DifferentialSpectrum


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Spectrum Difference.')
    parser.add_argument('spectrum1', help='First Spectrum')
    parser.add_argument('spectrum2', help="Second Spectrum")
    parser.add_argument('params', help="Orbital parameter file.")
    parser.add_argument("-p", '--plot', help="Plot differential")
    return parser.parse_args(args)


def load_spectrum(fname):
    data = fits.getdata(fname)
    header = fits.getheader(fname)

    return Spectrum(xaxis=data["wavelength"], flux=data["flux"], header=header)


def main(spectrum1, spectrum2, params=None, plot=True):
    spec1 = load_spectrum(spectrum1)
    spec2 = load_spectrum(spectrum2)

    # params = load/parse parameter file

    diff = DifferentialSpectrum(spec1, spec2, params=params)
    diff.barycentric_correct()

    if plot():
        diff.plot()
        plt.show()

    print("Done")


if __name__ == '__main__':
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    result = main(**opts)
    sys.exit(result)
