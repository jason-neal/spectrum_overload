# Spectrum Overload 
## Changelog

Upcoming release:



### 0.3.0
- Simplified overload operator function.
- Add instrument broadening
- Add indexing/slicing spectrum with [], (Returns new spectrum)
- Add more Type hints
- Handle all warnings as errors in testing.
- Fix test parameters to avoid invalid values. 
- Move test fixtures
- Test on python 3.7.
- Add makefile
- Add Appveyor and shippable configuration.
- Some general cleanup
- Update requirements to latest versions.


##### Depreciations:
- Drop support for python 2.7
  - Due xaxis and flux keywords required with "*,".
- Drop testing of python 3.4.
- Remove tox.ini


## 0.2.1 14/01/2018
  Patch try resolve pypi styling


## 0.2.0  14/01/2018

- Add hypothesis profiles for different test environments.
- Change header initialization from None to and empty dict
- Add normalization e.g.
	-  Spectrum.normalize("poly", degree=3)
- Add spectrum plot method.
- Standardize operations using a operator wrapper.
- Updated documetation.


## Version <0.2.0
Before october 2017
Many changes that were unrecorded in this change log...
Spectrum object with interpolation and overloaded operators.


Project started: September 2016
