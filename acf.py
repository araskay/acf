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
import fileutils

# the following two lines used to avoid envoking x display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def printhelp():
    print('Calculate the ACF FWHM for all the slices of a fMRI file and save individual FWHM values in the x and y directions as well as summary values to csv files.')
    print('Usage: acf.py --file <file name> [--ndiscard <n=0> --fit --iqrcoef <IQRcoef=1.5> --anomalythresh <AnomalyThresh=5>]')
    print('RUN FROM THE DIRECTORY WHERE YOU WANT TO HAVE THE CSV FIlES SAVED')
    print('--ndiscard <n=0>: number of frames at the beginning of the time-series to discard.')
    print('--fit: flag indicating FWHM to be calculated using the fitted ACF rather than the raw ACF. However, both raw and fitted ACFs are plotted regardless of whether --fit is used or not.')
    print('--iqrcoef <IQRcoef=1.5>: detect slices as outliers if FWHM > IQRcoef x IQR + Q3')
    print('--anomalythresh <AnomalyThresh=5>: detect slices as anomalous if FWHM > AnomalyThresh x min{FWHM}')

def acf2d(x):
    '''
    Calculate 2D autocorrelation function (ACF).
    Input:
           one slice
    Output:
           2D ACF
    '''
    F = ft.fft2(x)
    acf = ft.fftshift(ft.fft2(abs(F)**2))
    return acf


def acf3d(x):
    '''
    Calculate 3D autocorrelation function (ACF).
    Input:
            one volume
    Output:
            3D ACF
    '''
    F = ft.fftn(x)
    acf = ft.fftshift(ft.fftn(abs(F)**2))
    return acf


def ACF(x,a,b,c):
    '''
    Parmetric ACF function (1D) as sum of a Gaussian and an expontial function
    Inputs:
            x: point at which to calculate ACF
            a, b, c: ACF parameters
    Output:
            Value of the ACF function at x
    '''
    return a * np.exp(-x*x/(2.0*b*b)) + (1.0-a)*np.exp(-x/c)


def ACF_halfmaxshifted(x,a,b,c):
    return a * np.exp(-x*x/(2.0*b*b)) + (1.0-a)*np.exp(-x/c) - 0.5



def fwhm(acfx,dofit):
    '''
    Take a 1D ACF, fit a mixed Gaussian and exponential (if specified, as recommended by AFNI)
    and calculate FWHM for the fit.
    input:
            acfx: the ACF- 1D vector
            dofit: whether or not to the fit (boolean)
    output: (fwhm_pix, popt)
            fwhm_pix: FWHM in pixels
            popt: ACF function (defined above) parameters
    '''
    # use the following value if error in finding root encountered
    errorfwhm=50

    error=0 # 0: no error, 1: error in fitting, 2: error in finding root
    
    # normalize acf to [0,1]
    if max(acfx)>0:
        acfx = acfx/max(acfx)
    
    # crop the right side only
    acfx = acfx[np.argmax(acfx):-1]
    
    xdata = np.arange(len(acfx))
  
    # interpolate before curve fitting
    f = scipy.interpolate.interp1d(xdata,acfx, kind = "cubic")
    
    xnew = np.linspace(0,max(xdata),num=1000)

    # try to curve fit to the entire data points
    try:
        (popt, pcov) = optimization.curve_fit(ACF, xnew, f(xnew), p0=[0.5, 2, 2])
        #(popt, pcov) = optimization.curve_fit(ACF, xnew, f(xnew), p0=[0.5, 2, 2], bounds=(0,[1,np.inf,np.inf]))
    except RuntimeError:
        # if failed, try again with limited length of the ACF
        xprime = np.linspace(0,0.20*max(xdata),num=200)
        try:
            (popt, pcov) = optimization.curve_fit(ACF, xprime, f(xprime), p0=[0.5, 2, 2])
            #(popt, pcov) = optimization.curve_fit(ACF, xprime, f(xprime), p0=[0.5, 2, 2], bounds=(0,[1,np.inf,np.inf]))
        except RuntimeError:
            error=1
            popt=[]
    
    # if fitting was successful, use the fit, otherwise use the the original acf (interpolated) to find root (i.e., fwhm)
    if dofit and error==0: 
        f_halfmaxshifted = lambda x: ACF_halfmaxshifted(x,popt[0],popt[1],popt[2])
        
    else:
        f_halfmaxshifted = scipy.interpolate.interp1d(xdata,acfx-0.5, kind = "cubic")
    
    # find fwhm in pixels
    if np.sign(f_halfmaxshifted(0)) * np.sign(f_halfmaxshifted(len(acfx)/2)) < 0:
        fwhm_pix = 2 * optimization.brentq(f_halfmaxshifted,0,len(acfx)/2)    
    else:
        if np.sign(f_halfmaxshifted(0)) * np.sign(f_halfmaxshifted(len(acfx)-1)) < 0:
            fwhm_pix = 2 * optimization.brentq(f_halfmaxshifted,0,len(acfx)-1)
        else:
            error = 2
            return (errorfwhm, popt, error, f)
    
    return (fwhm_pix, popt, error, f)


def save_acf(acfx,popt,error,f,filename):
    '''
        save raw, interpolated, and fitted (if available) ACF upon error.
        Input:
                acfx: ACF function
                filename: file name to save the plots to
        Output:
                None- saves figures to disk
    '''
    ## plot ACFs
    # normalize acf to [0,1]
    if max(acfx)>0:
        acfx = acfx/max(acfx)
    
    # crop the right side only
    acfx = acfx[np.argmax(acfx):-1]
    xdata = np.arange(len(acfx))
    
    plt.figure(10)
    plt.plot(xdata,acfx)
    
    xnew = np.linspace(0,max(xdata),num=1000)
    
    plt.figure(10)
    plt.plot(xnew,f(xnew))
    
    if error!= 1:
        plt.figure(10)
        plt.plot(xnew,ACF(xnew,popt[0],popt[1],popt[2]))  
    
    if error == 1:
        plt.legend(['Original','Interpolated'])
    else:
        plt.legend(['Original','Interpolated','AFNI model'])
    
    plt.figure(10)
    plt.savefig(filename,dpi=600)
    plt.close() 


def main(input_args):
    input_file=''
    n_discard=0
    iqrcoef=1.5
    anomaly_thresh=5
    dofit=False

    # parse command-line arguments
    try:
        (opts,args) = getopt.getopt(input_args[1:],'h',['file=', 'help', 'ndiscard=','fit','iqrcoef=','anomalythresh='])
    except getopt.GetoptError:
        sys.exit()
    for (opt,arg) in opts:
        if opt in ('--file'):
            input_file=arg
        elif opt in ('--ndiscard'):
            n_discard=int(arg)
        elif opt in ('--fit'):
            dofit=True
        elif opt in ('--iqrcoef'):
            iqrcoef=float(arg)
        elif opt in ('--anomalythresh'):
            anomaly_thresh=float(arg)
        elif opt in ('-h','--help'):
            printhelp()
            sys.exit()
            
    if input_file=='':
        printhelp()
        sys.exit()

    errorfile = open('error.txt', 'a')

    # the following csv files store data per volume for each session
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

    # the following stores teq for each session
    csv_teq = open('teq.csv','a')

    # the following stores summary slice FWHM measures for each session
    csv_fwhm = open('acfFWHM.csv','a')

    img_nib=nb.load(input_file)
    img=img_nib.get_data()
    pixdim = img_nib.header.get_zooms()

    # now that the file is read, get rid of the path and the extension, and just use the base name to save the results:
    fmri_file = fileutils.namebase(input_file)

    sl_fwhmx = np.zeros((img.shape[2],img.shape[3]))
    sl_fwhmy = np.zeros((img.shape[2],img.shape[3]))

    for t in np.arange(img.shape[3]):
        for z in np.arange(img.shape[2]):
            sl=img[:,:,z,t]
            # calculate 2D ACF
            acf = acf2d(sl)

            # calculate FWHMx
            acfx = abs(acf[int(sl.shape[0]/2),])
            (fwhmx_pix, popt, error,f) = fwhm(acfx,dofit)
            sl_fwhmx[z,t] = fwhmx_pix * pixdim[0]
            
            ## plot ACFs
            if t >= n_discard:
                # normalize acf to [0,1]
                if max(acfx)>0:
                    acfx = acfx/max(acfx)
                # crop the right side only
                acfx = acfx[np.argmax(acfx):-1]
                xdata = np.arange(len(acfx))
                
                #plt.figure(11)
                #plt.plot(xdata,acfx)
                
                xnew = np.linspace(0,max(xdata),num=1000)
                
                plt.figure(12)
                plt.plot(xnew,f(xnew))
                
                if error!= 1:
                    plt.figure(13)
                    plt.plot(xnew,ACF(xnew,popt[0],popt[1],popt[2]))  

            # record errors
            if error != 0:
                errorfile.write(fmri_file+'  t='+str(t)+'  z='+str(z)+'  e='+str(error)+'  (FWHMx) \n')
                save_acf(acfx,popt,error,f,fmri_file+'_t'+str(t)+'_z'+str(z)+'_e'+str(error)+'_ACFx.png')
            
            # calculate FWHMy
            acfx = abs(acf[:,int(sl.shape[1]/2)])
            (fwhmx_pix, popt, error,f) = fwhm(acfx,dofit)
            sl_fwhmy[z,t] = fwhmx_pix * pixdim[1]
            
            ## plot ACFs
            if t >= n_discard:
                # normalize acf to [0,1]
                if max(acfx)>0:
                    acfx = acfx/max(acfx)
                # crop the right side only
                acfx = acfx[np.argmax(acfx):-1]
                xdata = np.arange(len(acfx))
                
                #plt.figure(21)
                #plt.plot(xdata,acfx)
                
                xnew = np.linspace(0,max(xdata),num=1000)
                
                plt.figure(22)
                plt.plot(xnew,f(xnew))
                
                if error!= 1:
                    plt.figure(23)
                    plt.plot(xnew,ACF(xnew,popt[0],popt[1],popt[2]))          
            
            # record errors
            if error != 0:
                errorfile.write(fmri_file+'  t='+str(t)+'  z='+str(z)+'  e='+str(error)+'  (FWHMy) \n')
                save_acf(acfx,popt,error,f,fmri_file+'_t'+str(t)+'_z'+str(z)+'_e'+str(error)+'_ACFy.png')  


    # find time to equillibrium
    # find t s.t. max(t:) == max(t+1:)
    max_sl_fwhmx = np.amax(sl_fwhmx,axis=0)
    max_sl_fwhmy = np.amax(sl_fwhmy,axis=0)
    t_eq=0
    while max(max_sl_fwhmx[t_eq:]) != max(max_sl_fwhmx[t_eq+1:]) and max(max_sl_fwhmy[t_eq:]) != max(max_sl_fwhmy[t_eq+1:]):
        t_eq+=1

    # crop to n_discard
    sl_fwhmx = sl_fwhmx[:,n_discard:]
    sl_fwhmy = sl_fwhmy[:,n_discard:]

    # anomalous slices
    anom_x = sl_fwhmx>anomaly_thresh*np.amin(sl_fwhmx)
    anom_y = sl_fwhmy>anomaly_thresh*np.amin(sl_fwhmy)

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

    # write summary measures to csv
    csv_fwhm.write(fmri_file)
    # summary measures in the x direction
    csv_fwhm.write(','+str(np.amax(sl_fwhmx)))
    csv_fwhm.write(','+str(np.amin(sl_fwhmx)))
    csv_fwhm.write(','+str(np.mean(sl_fwhmx)))
    csv_fwhm.write(','+str(np.median(sl_fwhmx)))
    csv_fwhm.write(','+str(np.percentile(sl_fwhmx,25)))
    csv_fwhm.write(','+str(np.percentile(sl_fwhmx,75)))
    csv_fwhm.write(','+str(np.std(sl_fwhmx)))
    # summary measures in the y direction
    csv_fwhm.write(','+str(np.amax(sl_fwhmy)))
    csv_fwhm.write(','+str(np.amin(sl_fwhmy)))
    csv_fwhm.write(','+str(np.mean(sl_fwhmy)))
    csv_fwhm.write(','+str(np.median(sl_fwhmy)))
    csv_fwhm.write(','+str(np.percentile(sl_fwhmy,25)))
    csv_fwhm.write(','+str(np.percentile(sl_fwhmy,75)))
    csv_fwhm.write(','+str(np.std(sl_fwhmy)))
    
    # fraction of anomalous slices per session
    # fracAnomaliesx,fracAnomaliesy
    csv_fwhm.write(','+str(np.sum(anom_x)/sl_fwhmx.shape[0]/sl_fwhmx.shape[1]))
    csv_fwhm.write(','+str(np.sum(anom_y)/sl_fwhmy.shape[0]/sl_fwhmy.shape[1]))

    # fraction of anomalous slices per volume
    # meanAnomalyPerVolx,stdAnomalyPerVolx
    csv_fwhm.write(','+str(np.mean(np.sum(anom_x,axis=0)/anom_x.shape[0])))
    csv_fwhm.write(','+str(np.std(np.sum(anom_x,axis=0)/anom_x.shape[0])))
    # meanAnomalyPerVoly,stdAnomalyPerVoly
    csv_fwhm.write(','+str(np.mean(np.sum(anom_y,axis=0)/anom_y.shape[0])))
    csv_fwhm.write(','+str(np.std(np.sum(anom_y,axis=0)/anom_y.shape[0])))

    csv_fwhm.write('\n')

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

    csv_fwhm.close()

    # save figures
    fileutils.createdir('acf')
    plt.figure(12)
    plt.savefig('acf/'+fmri_file+'_ACFx.png',dpi=600)
    plt.figure(13)
    plt.savefig('acf/'+fmri_file+'_ACFx_fit.png',dpi=600) 
    plt.figure(22)
    plt.savefig('acf/'+fmri_file+'_ACFy.png',dpi=600)
    plt.figure(23)
    plt.savefig('acf/'+fmri_file+'_ACFy_fit.png',dpi=600) 

    fileutils.createdir('fwhm')
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
    plt.savefig('fwhm/'+fmri_file+'_FWHMx.png',dpi=600)

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
    plt.savefig('fwhm/'+fmri_file+'_FWHMy.png',dpi=600)

    np.savetxt('fwhm/'+fmri_file+'_FWHMx.csv', sl_fwhmx.transpose(), delimiter=',')
    np.savetxt('fwhm/'+fmri_file+'_FWHMy.csv', sl_fwhmy.transpose(), delimiter=',')

    # find outliers wrt all slices in the sessions
    outlier_x = (sl_fwhmx > np.percentile(sl_fwhmx, 75) + iqrcoef * ss.iqr(sl_fwhmx)) | (sl_fwhmx < np.percentile(sl_fwhmx, 25) - iqrcoef * ss.iqr(sl_fwhmx))
    outlier_y = (sl_fwhmy > np.percentile(sl_fwhmy, 75) + iqrcoef * ss.iqr(sl_fwhmy)) | (sl_fwhmy < np.percentile(sl_fwhmy, 25) - iqrcoef * ss.iqr(sl_fwhmy))

    plt.figure(5)
    plt.imshow(outlier_x)
    plt.savefig('fwhm/'+fmri_file+'_FWHMx_outliers_'+str(iqrcoef)+'iqr.png',dpi=600)

    plt.figure(6)
    plt.imshow(outlier_y)
    plt.savefig('fwhm/'+fmri_file+'_FWHMy_outliers_'+str(iqrcoef)+'iqr.png',dpi=600)


if __name__ == '__main__':
    main(sys.argv)
