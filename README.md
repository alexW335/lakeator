# lakeator
A python package for Acoustic Source Localisation, with GUI and QGIS integration. 

Clone/refinement of the Locator package (github.com/alexW335/Locator) produced during my MSc; thesis on Acoustic Source Localisation available [here](https://mro.massey.ac.nz/handle/10179/15008).

## What is it?
The lakeator is a python class designed for use in acoustic source localisation. It contains functionality for three different algorithms, namely: the Generalised Cross-Correlation (GCC) algorithm [1], complete with the identity, PHAT, p-PHAT, SCOT, RIR, and HB processors [2]; the Multiple Signal Classification (MUSIC) algorithm for localisation of narrowband sources [3]; and the auto-focusing MUSIC (AF-MUSIC) algorithm for localisation of broadband signals [4]. 

## How do you use it?
The lakeator class can either be used inside a python script (see the main.py file, for example) or via the GUI with `python gui.py`. The GUI is designed to not require any programming knowledge to be able to use it. 

### Documentation
Documentation is available at https://lakeator.readthedocs.io/en/latest/

## Installation
To install dependencies with pipenv, use `pipenv install --ignore-pipfile`. The lakeator is dependent on the following packages, which will be installed:

* [PyQt5](http://python.org/pypi/PyQt5)
* [requests](http://python.org/pypi/requests)
* [numpy](http://python.org/pypi/numpy)
* [scipy](http://python.org/pypi/scipy)
* [matplotlib](http://python.org/pypi/matplotlib)
* [motionless](http://python.org/pypi/motionless)
* [pillow](http://python.org/pypi/Pillow)
* [pyproj](http://python.org/pypi/pyproj)


# References
[1] Knapp, C. H., & Carter, G. C. (1976). The Generalized Correlation Method for Estimation of Time Delay. IEEE Transactions on Acoustics, Speech, and Signal Processing, 24(4), 320–327. https://doi.org/10.1109/TASSP.1976.1162830 \
[2] Ritu, & Dhull, S. K. (2016). A Comparison of Generalized Cross-Correlation Methods for Time Delay Estimation. IUP Journal of Telecommunications, 8(4). \
[3] Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. IEEE Transactions on Antennas and Propagation, AP-34(3), 276–280. https://doi.org/10.1109/9780470544075.ch2 \
[4] Pal, P., & Vaidyanathan, P. P. (2009). A novel autofocusing approach for estimating directions-of-arrival of wideband signals. Conference Record - Asilomar Conference on Signals, Systems and Computers, 1663–1667. https://doi.org/10.1109/ACSSC.2009.5469796
