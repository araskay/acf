#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 10:38:13 2018

@author: aras
"""

import numpy as np, numpy.fft as ft
import nibabel as nb
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import scipy.optimize as optimization
import scipy.interpolate
import getopt, sys

def acf2d(x):
    F = ft.fft2(x)
    acf = ft.fftshift(ft.fft2(abs(F)**2))
    return acf


def acf3d(x):
    F = ft.fftn(x)
    acf = ft.fftshift(ft.fftn(abs(F)**2))
    return acf


def ACF(x,a,b,c):
    return a * np.exp(-x*x/(2.0*b*b)) + (1.0-a)*np.exp(-x/c)


def ACF_halfmaxshifted(x,a,b,c):
    return a * np.exp(-x*x/(2.0*b*b)) + (1.0-a)*np.exp(-x/c) - 0.5


# take a 1D ACF, fit a mixed Gaussian and exponential (as recommended by AFNI)
# and calculate FWHM for the fit
# input: acfx (1D vector)
# output: (fwhm_pix, popt)
#         fwhm_pix: FWHM in pixels
#         popt: ACF function (defined above) parameters
def fwhm(acfx):
    # normalize acf to [0,1]
    acfx = acfx/max(acfx)
    
    # crop the right side only
    acfx = acfx[np.argmax(acfx):-1]
    
    xdata = np.arange(len(acfx))
 
    # interpolate before curve fitting
    f = scipy.interpolate.interp1d(xdata,acfx, kind = "cubic")
    
    xnew = np.linspace(0,max(xdata),num=10000)
   
    (popt, pcov) = optimization.curve_fit(ACF, xnew, f(xnew), [0.5, 2, 2])
    
    # fwhm in pixels
    fwhm_pix = 2 * optimization.brentq(ACF_halfmaxshifted,0,len(acfx),args=(popt[0],popt[1],popt[2]))    

    return (fwhm_pix, popt)


###############################################################################   
# Description - This function takes in a 1D ACF cross-section vector, fits a 
    # weighted univariate spline, and calculates the FWHM
# Args - 1D vector (ACF cross-section),
# Returns - FWHM value in pixels
############################################################################### 
def aditi_fwhm(y):
    a=0.0
    b=15.0
    y = y[int(len(y)/2):]
    x = np.linspace(0,len(y)-1,len(y))
    w = np.ones(len(x))
    w[0:3]=[4.0,3.0,2.0] 
    ### if it does not go below zero, return large number
    if any(n<0 for n in (y-np.max(y)/2)) == False:
        return len(x)*4.0      
    if len(x)==len(y):
        interp = scipy.interpolate.UnivariateSpline(x,y-np.max(y)/2,s=0.0,k=5.0,w=w)
    ### find first negative value to assign to value 'b'
    b = next((i for i, v in enumerate(interp(x)) if v < 0.0), 15.0)
    roots = scipy.optimize.brentq(interp,a,b)
    return (roots*2.0)/np.sqrt(2.0), interp

def printhelp():
    print('Usage: plot_acf -f <file name>')

fmri_file=''
# parse command-line arguments
try:
    (opts,args) = getopt.getopt(sys.argv[1:],'hf:',['file=', 'help'])
except getopt.GetoptError:
    sys.exit()
for (opt,arg) in opts:
    if opt in ('-f','--file'):
        fmri_file=arg
    elif opt in ('-h','--help'):
        printhelp()
        
if fmri_file=='':
    printhelp()
    sys.exit()
        

img_nib=nb.load(fmri_file)
img=img_nib.get_data()
pixdim = img_nib.header.get_zooms()

sl=img[:,:,int(img.shape[2]/2),1]
# calculate 2D ACF
acf = acf2d(sl)
# calculate FWHMx
acfx = abs(acf[int(sl.shape[0]/2),])
(fwhmx_pix, poptx) = fwhm(acfx)

fwhmx=fwhmx_pix * pixdim[0]

plt.figure(1)

plt.plot(acfx[int(img.shape[0]/2):-1]/max(acfx))
plt.hold(True)

x = np.linspace(0,img.shape[0]/2,num=10000)
y = ACF(x,poptx[0],poptx[1],poptx[2])

plt.plot(x,y)

fwhmx_pix_compare, interp_compare = aditi_fwhm(acfx)
fwhmx_compare = fwhmx_pix_compare * pixdim[0]
y=interp_compare(x)
plt.plot(x,y/max(y)/2+0.5)

plt.title('ACF')
plt.legend(['Original','AFNI model', 'Spline interpolate'])

plt.savefig(fmri_file+'_acf.png')

# calculate FWHMy
acfy = abs(acf[:,int(sl.shape[1]/2)])
(fwhmy_pix, popty) = fwhm(acfy)
fwhmy=fwhmy_pix * pixdim[1]










