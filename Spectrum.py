#!/usr/bin/python
from __future__ import print_function, division

# Spectrum Class

# Begun August 2016
# Jason Neal


class Spectrum:
    """ Spectrum class represents and manipulates astronomical spectra. """
    
    def __init__(self, pixel=[], flux=[], wavelength=[]):
        """ Create a empty spectra """
        self.pixel = pixel
        self.flux = flux
        self.wavelength = wavelength




# Try using Spectrum 
#if __name__ == __main__:

x = [1,2,3,4,5,6]
y = [1,1,0.9,0.95,1,1]

test = Spectrum()
print(test)
print(test.pixel, test.flux)
test.pixel = x
test.flux = y
print(test)
print(test.pixel, test.flux, test.wavelength)

test2 = Spectrum(x, flux=y)
print(test2)
print(test2.pixel, test.flux)
print(test2.wavelength)