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
import matplotlib
matplotlib.use('Agg')
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
        (popt, pcov) = optimization.curve_fit(ACF, xnew, f(xnew), p0=[0.5, 2, 2], bounds=(0,[1,1000000,1000000]))
    except RuntimeError:
        # try again with limited length of the ACF
        xprime = np.linspace(0,0.20*max(xdata),num=200)
        try:
            #(popt, pcov) = optimization.curve_fit(ACF, xprime, f(xprime), [0.5, 2, 2])
            (popt, pcov) = optimization.curve_fit(ACF, xprime, f(xprime), p0=[0.5, 2, 2], bounds=(0,[1,1000000000,1000000000]))
        except:
            error=1
            popt=[]
    
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
            return (errorfwhm, popt, error)

    return (fwhm_pix, popt, error)


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

def printhelp():
    print('Usage: acf.py -f <file name> [--ndiscard <n=0>]')

fmri_file=''
n_discard=0

# parse command-line arguments
try:
    (opts,args) = getopt.getopt(sys.argv[1:],'hf:',['file=', 'help', 'ndiscard='])
except getopt.GetoptError:
    sys.exit()
for (opt,arg) in opts:
    if opt in ('-f','--file'):
        fmri_file=arg
    elif opt in ('--ndiscard'):
        n_discard=int(arg)
    elif opt in ('-h','--help'):
        printhelp()
        
if fmri_file=='':
    printhelp()
    sys.exit()

errorfile = open('error.txt', 'a')
 
csv_maxFWHMx = open('maxFWHMx.csv', 'a')
csv_minFWHMx = open('minFWHMx.csv', 'a')
csv_medFWHMx = open('medFWHMx.csv', 'a')
csv_q1FWHMx = open('q1FWHMx.csv', 'a')
csv_q3FWHMx = open('q3FWHMx.csv', 'a')
csv_meanFWHMx = open('meanFWHMx.csv', 'a')
csv_stdFWHMx = open('stdFWHMx.csv', 'a')

csv_maxFWHMy = open('maxFWHMy.csv', 'a')
csv_minFWHMy = open('minFWHMy.csv', 'a')
csv_medFWHMy = open('medFWHMy.csv', 'a')
csv_q1FWHMy = open('q1FWHMy.csv', 'a')
csv_q3FWHMy = open('q3FWHMy.csv', 'a')
csv_meanFWHMy = open('meanFWHMy.csv', 'a')
csv_stdFWHMy = open('stdFWHMy.csv', 'a')

csv_teq = open('teq.csv','a')

img_nib=nb.load(fmri_file)
img=img_nib.get_data()
pixdim = img_nib.header.get_zooms()

# find time to equillibrium
sl_fwhmx = np.zeros((img.shape[2],img.shape[3]))
sl_fwhmy = np.zeros((img.shape[2],img.shape[3]))
for t in np.arange(img.shape[3]):
    for z in np.arange(img.shape[2]):
        sl=img[:,:,z,t]
        # calculate 2D ACF
        acf = acf2d(sl)
        # calculate FWHMx
        acfx = abs(acf[int(sl.shape[0]/2),])
        (fwhmx_pix, poptx, error) = fwhm(acfx)
        sl_fwhmx[z,t] = fwhmx_pix * pixdim[0]    
        # calculate FWHMy
        acfy = abs(acf[:,int(sl.shape[1]/2)])
        (fwhmy_pix, popty, error) = fwhm(acfy)
        sl_fwhmy[z,t] = fwhmy_pix * pixdim[1]
# find t s.t. max(t:) == max(t+1:)
max_sl_fwhmx = np.amax(sl_fwhmx,axis=0)
max_sl_fwhmy = np.amax(sl_fwhmy,axis=0)
t_eq=0
while max(max_sl_fwhmx[t_eq:]) != max(max_sl_fwhmx[t_eq+1:]) and max(max_sl_fwhmy[t_eq:]) != max(max_sl_fwhmy[t_eq+1:]):
    t_eq+=1
###################################

sl_fwhmx = np.zeros((img.shape[2],img.shape[3]))
sl_fwhmy = np.zeros((img.shape[2],img.shape[3]))

for t in np.arange(n_discard,img.shape[3]):
    for z in np.arange(img.shape[2]):
        sl=img[:,:,z,t]
        # calculate 2D ACF
        acf = acf2d(sl)
        # calculate FWHMx
        acfx = abs(acf[int(sl.shape[0]/2),])
        (fwhmx_pix, poptx, error) = fwhm(acfx)
        sl_fwhmx[z,t] = fwhmx_pix * pixdim[0]
        
        if error == 0:
            plt.figure(1)
            x = np.linspace(0,img.shape[0]/2,num=10000)
            y = ACF(x,poptx[0],poptx[1],poptx[2])
            plt.plot(x,y)
        else:
            errorfile.write(fmri_file+'  t='+str(t)+'  z='+str(z)+'  e='+str(error)+'  (FWHMx) \n')
            save_acf(acfx,img,fmri_file+'_t'+str(t)+'_z'+str(z)+'_e'+str(error)+'_ACFx.png')
        
        # calculate FWHMy
        acfy = abs(acf[:,int(sl.shape[1]/2)])
        (fwhmy_pix, popty, error) = fwhm(acfy)
        sl_fwhmy[z,t] = fwhmy_pix * pixdim[1]
        if error == 0:
            plt.figure(2)
            x = np.linspace(0,img.shape[1]/2,num=10000)
            y = ACF(x,popty[0],popty[1],popty[2])
            plt.plot(x,y)            
        else:
            errorfile.write(fmri_file+'  t='+str(t)+'  z='+str(z)+'  e='+str(error)+'  (FWHMy) \n')
            save_acf(acfy,img,fmri_file+' _t'+str(t)+'_z'+str(z)+'_e'+str(error)+'_ACFy.png')  

sl_fwhmx = sl_fwhmx[:,n_discard:]
sl_fwhmy = sl_fwhmy[:,n_discard:]

# write to csv
csv_maxFWHMx.write(fmri_file+','+','.join(str(x) for x in np.amax(sl_fwhmx,axis=0))+'\n')
csv_minFWHMx.write(fmri_file+','+','.join(str(x) for x in np.amin(sl_fwhmx,axis=0))+'\n')
csv_medFWHMx.write(fmri_file+','+','.join(str(x) for x in np.median(sl_fwhmx,axis=0))+'\n')
csv_q1FWHMx.write(fmri_file+','+','.join(str(x) for x in np.percentile(sl_fwhmx,25,axis=0))+'\n')
csv_q3FWHMx.write(fmri_file+','+','.join(str(x) for x in np.percentile(sl_fwhmx,75,axis=0))+'\n')
csv_meanFWHMx.write(fmri_file+','+','.join(str(x) for x in np.mean(sl_fwhmx,axis=0))+'\n')
csv_stdFWHMx.write(fmri_file+','+','.join(str(x) for x in np.std(sl_fwhmx,axis=0))+'\n')

csv_maxFWHMy.write(fmri_file+','+','.join(str(x) for x in np.amax(sl_fwhmy,axis=0))+'\n')
csv_minFWHMy.write(fmri_file+','+','.join(str(x) for x in np.amin(sl_fwhmy,axis=0))+'\n')
csv_medFWHMy.write(fmri_file+','+','.join(str(x) for x in np.median(sl_fwhmy,axis=0))+'\n')
csv_q1FWHMy.write(fmri_file+','+','.join(str(x) for x in np.percentile(sl_fwhmy,25,axis=0))+'\n')
csv_q3FWHMy.write(fmri_file+','+','.join(str(x) for x in np.percentile(sl_fwhmy,75,axis=0))+'\n')
csv_meanFWHMy.write(fmri_file+','+','.join(str(x) for x in np.mean(sl_fwhmy,axis=0))+'\n')
csv_stdFWHMy.write(fmri_file+','+','.join(str(x) for x in np.std(sl_fwhmy,axis=0))+'\n')

csv_teq.write(fmri_file+','+str(t_eq)+'\n')

csv_maxFWHMx.close()
csv_minFWHMx.close()
csv_medFWHMx.close()
csv_q1FWHMx.close()
csv_q3FWHMx.close()              
csv_meanFWHMx.close()
csv_stdFWHMx.close()

csv_maxFWHMy.close()
csv_minFWHMy.close()
csv_medFWHMy.close()
csv_q1FWHMy.close()
csv_q3FWHMy.close()              
csv_meanFWHMy.close()
csv_stdFWHMy.close()            

errorfile.close()

csv_teq.close()

# save figures
plt.figure(1)
plt.savefig(fmri_file+'_ACFx.png')
plt.figure(2)
plt.savefig(fmri_file+'_ACFy.png') 

plt.figure(3)
plt.plot(np.amax(sl_fwhmx,axis=0))
plt.plot(np.amin(sl_fwhmx,axis=0))
plt.plot(np.median(sl_fwhmx,axis=0))
plt.plot(np.percentile(sl_fwhmx,25,axis=0))
plt.plot(np.percentile(sl_fwhmx,75,axis=0))
plt.plot(np.mean(sl_fwhmx,axis=0))
plt.plot(np.std(sl_fwhmx,axis=0))
plt.legend(['Max','Min','Med','Q1','Q3','Mean','STD'])
plt.xlabel('Frame number')
plt.ylabel('FWHM (mm)')
plt.savefig(fmri_file+'_FWHMx.png')

plt.figure(4)
plt.plot(np.amax(sl_fwhmy,axis=0))
plt.plot(np.amin(sl_fwhmy,axis=0))
plt.plot(np.median(sl_fwhmy,axis=0))
plt.plot(np.percentile(sl_fwhmy,25,axis=0))
plt.plot(np.percentile(sl_fwhmy,75,axis=0))
plt.plot(np.mean(sl_fwhmy,axis=0))
plt.plot(np.std(sl_fwhmy,axis=0))
plt.legend(['Max','Min','Med','Q1','Q3','Mean','STD'])
plt.xlabel('Frame number')
plt.ylabel('FWHM (mm)')
plt.savefig(fmri_file+'_FWHMy.png')

# find outliers wrt all slices in the sessions
outlier_x = sl_fwhmx > np.percentile(sl_fwhmx, 75) + 1.5 * ss.iqr(sl_fwhmx)
outlier_y = sl_fwhmy > np.percentile(sl_fwhmy, 75) + 1.5 * ss.iqr(sl_fwhmy)

plt.figure(5)
plt.imshow(outlier_x)
plt.savefig(fmri_file+'_FWHMx_outliers.png')

plt.figure(6)
plt.imshow(outlier_y)
plt.savefig(fmri_file+'_FWHMy_outliers.png')

# find outliers wrt slices in each volume
outlier_x = np.zeros(sl_fwhmx.shape)
outlier_y = np.zeros(sl_fwhmy.shape)

for t in np.arange(sl_fwhmx.shape[1]):
    outlier_x[:,t] = sl_fwhmx[:,t] > np.percentile(sl_fwhmx[:,t], 75) + 1.5 * ss.iqr(sl_fwhmx[:,t])
    outlier_y[:,t] = sl_fwhmy[:,t] > np.percentile(sl_fwhmy[:,t], 75) + 1.5 * ss.iqr(sl_fwhmy[:,t])
    
plt.figure(7)
plt.imshow(outlier_x)
plt.savefig(fmri_file+'_FWHMx_outliers_vol.png')

plt.figure(8)
plt.imshow(outlier_y)
plt.savefig(fmri_file+'_FWHMy_outliers_vol.png')