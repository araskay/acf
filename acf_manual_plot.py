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


# the following function copied from Aditi's script. Used only for plotting
        # for comparison
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
    
    interp = scipy.interpolate.UnivariateSpline(x,y-np.max(y)/2,s=0.0,k=5.0,w=w)
    
    ### if it does not go below zero, return large number
    if any(n<0 for n in (y-np.max(y)/2)) == False:
        return len(x)*4.0, interp
    ### find first negative value to assign to value 'b'
    b = next((i for i, v in enumerate(interp(x)) if v < 0.0), 15.0)
    roots = scipy.optimize.brentq(interp,a,b)
    return (roots*2.0)/np.sqrt(2.0), interp


# take a 1D ACF, fit a mixed Gaussian and exponential (as recommended by AFNI)
# and calculate FWHM for the fit
# input: acfx (1D vector)
# output: (fwhm_pix, popt)
#         fwhm_pix: FWHM in pixels
#         popt: ACF function (defined above) parameters
def fwhm(acfx):
    
    # use the following value if any error encountered
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
        #(popt, pcov) = optimization.curve_fit(ACF, xnew, f(xnew), p0=[0.5, 2, 2], bounds=(0,[1,1000000,1000000]))
        (popt, pcov) = optimization.curve_fit(ACF, xnew, f(xnew), p0=[0.5, 2, 2])
    except RuntimeError:
        # try again with limited length of the ACF
        xprime = np.linspace(0,0.20*max(xdata),num=200)
        try:
            #(popt, pcov) = optimization.curve_fit(ACF, xprime, f(xprime), p0=[0.5, 2, 2], bounds=(0,[1,1000000,1000000]))
            (popt, pcov) = optimization.curve_fit(ACF, xprime, f(xprime), p0=[0.5, 2, 2])
        except:
            error=1
            return (errorfwhm, [], error)
    
    # find fwhm in pixels
    if np.sign(ACF_halfmaxshifted(0,popt[0],popt[1],popt[2])) * np.sign(ACF_halfmaxshifted(len(acfx)/2,popt[0],popt[1],popt[2])) < 0:
        fwhm_pix = 2 * optimization.brentq(ACF_halfmaxshifted,0,len(acfx)/2,args=(popt[0],popt[1],popt[2]))    
    else:
        if np.sign(ACF_halfmaxshifted(0,popt[0],popt[1],popt[2])) * np.sign(ACF_halfmaxshifted(len(acfx),popt[0],popt[1],popt[2])) < 0:
            fwhm_pix = 2 * optimization.brentq(ACF_halfmaxshifted,0,len(acfx),args=(popt[0],popt[1],popt[2]))
        else:
            error = 2
            return (errorfwhm, popt, error)

    return (fwhm_pix, popt, error)




# save raw, interpolated, and fitted (if available) ACF upon error
def save_acf(acfx,img,filename):
    plt.figure(10)
    
    # plot original acf
    plt.plot(acfx[int(img.shape[0]/2):-1]/max(acfx))
    
    x = np.linspace(0,img.shape[0]/2,num=10000)
    
    # plot AFNI model (Gaussian plus exponential)
    (fwhmx_pix, poptx, error) = fwhm(acfx)
    if error!=1:
        y = ACF(x,poptx[0],poptx[1],poptx[2])
        plt.plot(x,y)
    
    # plot Aditi's model (spline interpolation)
    fwhmx_pix_compare, interp_compare = aditi_fwhm(acfx)
    y=interp_compare(x)
    plt.plot(x,y/max(y)/2+0.5)
    
    plt.title('ACF')
    
    if error!=1:
        plt.legend(['Original','AFNI model', 'Spline interpolate'])
    else:
        plt.legend(['Original', 'Spline interpolate'])
    
    plt.figure(10)
    plt.savefig(filename)
    plt.close() 


fmri_file='detrend_WEU01_PHA_FBN1391_0010.nii'


img_nib=nb.load(fmri_file)
img=img_nib.get_data()
pixdim = img_nib.header.get_zooms()

t=50
z= 32
sl=img[:,:,z,t]

acf = acf2d(sl)
# calculate FWHMx
acfx = abs(acf[int(sl.shape[0]/2),])
(fwhmx_pix, poptx, error) = fwhm(acfx)
        
plt.figure(1)
x = np.linspace(-img.shape[0]/10,img.shape[0]/10,num=10000)
y = ACF(abs(x),poptx[0],poptx[1],poptx[2])
plt.plot(x*pixdim[0],y)
plt.xlabel('x (mm)')
plt.ylabel('ACF')
#plt.show()
plt.savefig('acf_AFNImodel_sample.png')


