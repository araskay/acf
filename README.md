# Autocorrelation function (ACF) analysis for fMRI
This repository provides tools to calculate the autocorrelation function (ACF) for each slice in a fMRI acquisition. The full width at half maximum (FWHM) of the ACF is calculated as a measure of spatial resolution. FWHM values and their summary measures are saved in csv files.

## Getting started
*acf.py* calculates the ACF FWHM for all the slices of a fMRI file and save individual FWHM values in the x and y directions as well as summary values to csv files.

## Installation
Clone or download this repository to a directory of your choice. Make sure all the prerequisites are installed and accessible. (See section Prerequisites.)

**To use parallel processing features**, add the installation directory to your system path. For example, if the repository is cloned to */home/user/acf/* on a linux system, add the following line to your *.bashrc* or *.bash_profile*:
```
export PATH=$PATH:/home/user/acf
```

## Prerequisites
ACF requires python3.

The required libraries are listed in *requirements.txt*. These libraries can be installed using *pip*:
```
pip install -r requirements.txt
```

**To use parallel processing features**, the [fmri_pipeline](https://github.com/kayvanrad/fmri_pipeline) tool should be installed on the machine/server. Please refer to the [fmri_pipeline](https://github.com/kayvanrad/fmri_pipeline) documentation for installation instructions.

## Author
[Aras Kayvanrad](https://www.linkedin.com/in/kayvanrad/)

**Related repos:** [fmri_pipeline](https://github.com/kayvanrad/fmri_pipeline), [fbirnQA](https://github.com/kayvanrad/fbirnQA), [phantomQA](https://github.com/kayvanrad/phantomQA)

