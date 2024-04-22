# magplots
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10703227.svg)](https://doi.org/10.5281/zenodo.10703227)
[![PyPI version](https://badge.fury.io/py/magplots.svg)](https://badge.fury.io/py/magplots)

A python library to pull data and make plots from ground magnetometers. Emphasis on plots for comparing magnetometers at geomagnetic conjugate points.


<!-- Click here to run example code: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KCollins/magplots/HEAD?labpath=Examples.ipynb) -->

## Installation
`pip install magplots`
Make sure that you have a folder in your directory named `/output`. This is where plots and dataframes will be saved.

## MWE
You can check all the component functions by running the `magall()` function with default values.
```
from magplots.magFunctions import * # packaged version
# from magFunctions import * # local version
magall(is_verbose = True)
```

# Functions
Type `help('magFunctions')` for function documentation.

To pull data from Tromsø Geophysical Observatory, save a password locally in a file named `tgopw.txt`. (You will have to obtain the password from TGO.) If no such file exists, data is automatically pulled from CDAWeb.

## Example Plots
### Time domain plot: 
This plot is produced with the `magplot()` function:
![timedomain](output/Example_TimeDomain_2016-01-24_Bx.png)

### Spectrogram: 
This plot is produced with the `magspect()` function:
![spectrogram](output/Example_PowerSpectrum_2016-01-24_Bx.png)

### Wave Power Plot: 
This plot is produced with the `wavefig()` function, which calls `wavepwr()` to calculate the wave power for each station within a given frequency range. By default the frequency range used is the Pc5 range, 1.667 to 6.667 mHz (150-600s). For a given timespan, the integrated wave power is plotted versus the latitude of the stations, with different colors for Arctic and Antarctic magnetometers, in order to illustrate hemispheric asymmetry. This approach is based on Figure 4 of [Pilipenko 2021](https://www.doi.org/10.1029/2020JA028048).
![waveplot](output/Example_WavePower_2016-01-24_Bx.png)
