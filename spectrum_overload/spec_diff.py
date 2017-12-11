import argparse
import sys

import matplotlib.pyplot as plt
from astropy.io import fits
from observationtools import RV
from observationtools.utils.parse import parse_paramfile

from spectrum_overload import Spectrum, DifferentialSpectrum


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Spectrum Difference.')
    parser.add_argument('spectrum1', help='First Spectrum')
    parser.add_argument('spectrum2', help="Second Spectrum")
    parser.add_argument('param_file', help="Orbital parameter file.")
    parser.add_argument("-p", '--plot', help="Plot differential", action="store_true")
    return parser.parse_args(args)


def load_spectrum(fname):
    data = fits.getdata(fname)
    header = fits.getheader(fname)

    return Spectrum(xaxis=data["wavelength"], flux=data["flux"], header=header)


def main(spectrum1, spectrum2, param_file=None, plot=True):
    spec1 = load_spectrum(spectrum1)
    # for key, value in spec1.header.items():
    #    print(key, value)
    spec2 = load_spectrum(spectrum2)
    # params = load/parse parameter file
    params = parse_paramfile(param_file)
    print("params", params)
    diff = DifferentialSpectrum(spec1, spec2, params=params)
    diff.barycentric_correct()

    host_rv = RV.from_file(param_file)
    if plot:
        fig, axes = plt.subplots(2, 1, sharex=True)
        spec1.plot(axis=axes[0], label="spec1")
        spec2.plot(axis=axes[0], label="spec2")
        plt.title(spec1.header["OBJECT"])
        plt.ylabel(r"Flux")
        diff.plot(axis=axes[1], label="difference")
        plt.ylabel(r"\Delta Flux")
        plt.xlabel("Wavelength")

        plt.legend()
        plt.show()

    print("Done")


if __name__ == '__main__':
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    result = main(**opts)
    sys.exit(result)
