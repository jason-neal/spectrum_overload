#!\bin\bash
# automatic install and then test script for Spectrum
# install package 
python setup.py install
# Test package
py.test