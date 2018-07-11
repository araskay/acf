#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:20:36 2018

@author: mkayvanrad
"""

import numpy as np, numpy.fft as ft
import nibabel as nb
import scipy.optimize as optimization
import scipy.interpolate, scipy.stats as ss
import getopt,sys

# the following two lines used to avoid envoking x display
import matplotlib.pyplot as plt

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
    # use the following value if error in finding root encountered
    errorfwhm=50

    error=0 # 0: no error, 1: error in fitting, 2: error in finding root
    
    # normalize acf to [0,1]
    acfx = acfx/max(acfx)
    
    # crop the right side only
    acfx = acfx[np.argmax(acfx):-1]
    
    xdata = np.arange(len(acfx))
  
    # interpolate before curve fitting
    f = scipy.interpolate.interp1d(xdata,acfx, kind = "cubic")
    
    xnew = np.linspace(0,max(xdata),num=1000)

    try:
        #(popt, pcov) = optimization.curve_fit(ACF, xnew, f(xnew), [0.5, 2, 2])
        (popt, pcov) = optimization.curve_fit(ACF, xnew, f(xnew), p0=[0.5, 2, 2], bounds=(0,[1,1e32,1e32]))
    except RuntimeError:
        # try again with limited length of the ACF
        xprime = np.linspace(0,0.20*max(xdata),num=200)
        try:
            #(popt, pcov) = optimization.curve_fit(ACF, xprime, f(xprime), [0.5, 2, 2])
            (popt, pcov) = optimization.curve_fit(ACF, xprime, f(xprime), p0=[0.5, 2, 2], bounds=(0,[1,1e32,1e32]))
        except RuntimeError:
            error=1
            popt=[]
    
    # if fitting was successful, use the fit, otherwise use the the original acf (interpolated) to find root (i.e., fwhm)
    if error==0: 
        f_halfmaxshifted = lambda x: ACF_halfmaxshifted(x,popt[0],popt[1],popt[2])
        
    else:
        f_halfmaxshifted = scipy.interpolate.interp1d(xdata,acfx-0.5, kind = "cubic")
    
    # find fwhm in pixels
    if np.sign(f_halfmaxshifted(0)) * np.sign(f_halfmaxshifted(len(acfx)/2)) < 0:
        fwhm_pix = 2 * optimization.brentq(f_halfmaxshifted,0,len(acfx)/2)    
    else:
        if np.sign(f_halfmaxshifted(0)) * np.sign(f_halfmaxshifted(len(acfx))) < 0:
            fwhm_pix = 2 * optimization.brentq(f_halfmaxshifted,0,len(acfx))
        else:
            error = 2
            return (errorfwhm, popt, error, f)
    
    return (fwhm_pix, popt, error, f)

def printhelp():
    print('Usage: acf.py -f <file name> -t <time frame> -s <slice>')

fmri_file=''
t_in=''
s_in=''

# parse command-line arguments
try:
    (opts,args) = getopt.getopt(sys.argv[1:],'hf:t:s:')
except getopt.GetoptError:
    sys.exit()
for (opt,arg) in opts:
    if opt in ('-f'):
        fmri_file=arg
    elif opt in ('-t'):
        t_in=arg
    elif opt in ('-s'):
        s_in=arg
    elif opt in ('-h'):
        printhelp()
        sys.exit()
        
if fmri_file=='' or t_in=='':
    printhelp()
    sys.exit()
    
t=int(t_in)

img_nib=nb.load(fmri_file)
img=img_nib.get_data()
pixdim = img_nib.header.get_zooms()

if len(s_in)>0:
    s=int(s_in)
    sl=img[:,:,s,t]
    # calculate 2D ACF
    acf = acf2d(sl)
    # calculate FWHMx
    acfx = abs(acf[int(sl.shape[0]/2),])
    (fwhmx_pix, popt, error,f) = fwhm(acfx)
    sl_fwhmx = fwhmx_pix * pixdim[0]    

    print(s,sl_fwhmx)
    
    ## plot ACFs
    # normalize acf to [0,1]
    acfx = acfx/max(acfx)
    
    # crop the right side only
    acfx = acfx[np.argmax(acfx):-1]
    xdata = np.arange(len(acfx))
    
    plt.figure(2)
    plt.plot(xdata,acfx)
    
    xnew = np.linspace(0,max(xdata),num=1000)
    
    plt.figure(2)
    plt.plot(xnew,f(xnew))
    
    if error!= 1:
        plt.figure(2)
        plt.plot(xnew,ACF(xnew,popt[0],popt[1],popt[2]))  
    
    if error == 1:
        plt.legend(['Original','Interpolated'])
    else:
        plt.legend(['Original','Interpolated','AFNI model'])        
    plt.show()
        
else:
    for s in np.arange(img.shape[2]):
        sl=img[:,:,s,t]
        # calculate 2D ACF
        acf = acf2d(sl)
        # calculate FWHMx
        acfx = abs(acf[int(sl.shape[0]/2),])
        (fwhmx_pix, popt, error,f) = fwhm(acfx)
        sl_fwhmx = fwhmx_pix * pixdim[0]    

        print(s,sl_fwhmx)

        ## plot ACFs
        # normalize acf to [0,1]
        acfx = acfx/max(acfx)
        
        # crop the right side only
        acfx = acfx[np.argmax(acfx):-1]
        xdata = np.arange(len(acfx))
        
        plt.figure(2)
        plt.plot(xdata,acfx)
        
        xnew = np.linspace(0,max(xdata),num=1000)
        
        plt.figure(2)
        plt.plot(xnew,f(xnew))
        
        if error!= 1:
                plt.figure(2)
                plt.plot(xnew,ACF(xnew,popt[0],popt[1],popt[2]))  
        
        if error == 1:
            plt.legend(['Original','Interpolated'])
        else:
            plt.legend(['Original','Interpolated','AFNI model'])        
        plt.show()
    
    