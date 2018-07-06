# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 16:43:07 2016

@author: achem
"""

import sys, getopt
from scipy import fftpack
import numpy as np
import nibabel
import scipy.interpolate as interpolate
import collections
import glob
import os
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from skimage.feature import peak_local_max
from skimage.transform import hough_circle
from skimage import filter
from scipy.optimize import curve_fit
import scipy.optimize

###############################################################################   
# Description - This function takes in a slice (2D matrix) and calculates
#   its ACF in the frequency domain
# Args - one slice
# Returns - 2D ACF matrix
###############################################################################  
def acffunc(sl):
    F = fftpack.fft2(sl)
    temp = F * np.conjugate(F)
    acf = fftpack.fftshift(fftpack.ifft2(temp))
    acfr=acf.real
    acfr=acfr/np.max(np.max(acfr))
    return acfr
    
    
###############################################################################   
# Description - This function takes in a 1D ACF cross-section vector, fits a 
    # weighted univariate spline, and calculates the FWHM
# Args - 1D vector (ACF cross-section),
# Returns - FWHM value in pixels
############################################################################### 
def fwhm(y):
    a=0.0
    b=15.0
    y = y[len(y)/2.0:]
    x = np.linspace(0,len(y)-1,len(y))
    w = np.ones(len(x))
    w[0:3]=[4.0,3.0,2.0] 
    ### if it does not go below zero, return large number
    if any(n<0 for n in (y-np.max(y)/2)) == False:
        return len(x)*4.0      
    if len(x)==len(y):
        interp = interpolate.UnivariateSpline(x,y-np.max(y)/2,s=0.0,k=5.0,w=w)
    ### find first negative value to assign to value 'b'
    b = next((i for i, v in enumerate(interp(x)) if v < 0.0), 15.0)
    roots = scipy.optimize.brentq(interp,a,b)
    return (roots*2.0)/np.sqrt(2.0)   
    
    
###############################################################################   
# Description - This function takes in the centre of a circular space and
    # coordintes in that space, and rotates it at a given angle in degrees
# Args - x & y coordinates, x & y coordinates of centre, rotation angle (deg)
# Returns - coordinates of a location in the new space after rotation
###############################################################################     
def rotated_rectangle(x,y,xcen,ycen,theta):
    theta=theta*np.pi/180
    tempx = x-xcen
    tempy = y-ycen
    x_rot = tempx*np.cos(theta) - tempy*np.sin(theta)
    y_rot = tempx*np.sin(theta) + tempy*np.cos(theta)
    x_rot = x_rot+xcen
    y_rot = y_rot+ycen
    return np.round(x_rot),np.round(y_rot)

###############################################################################   
# Description - The main function creates an ACF results report and summary 
    # values for an input fMRI time series
# Args - path to fMRI time series
############################################################################### 
def main(f):
    # number of volumes to exclude from the beginning and end
    acqrem = 10
    acqremend=0
#    analysis='detrendedAFNIsmoothed7mm'
    analysis='AFNIdetrended'
    # path to pre-processed data
    scan_dir = '/home/achemparathy/PHANTOM_DATA/PHA_'+analysis+'/'
    # path to raw data
    rawscan_dir = '/home/achemparathy/PHANTOM_DATA/raw_NIFTIs/'
    # name of summary values results output csv
    suffix = '_10volsrem_40x10' 
    output_filename='/home/achemparathy/PHANTOM_DATA/PHA_'+analysis+'/acfsummaryvals_21082017_'+analysis+suffix+'.csv'
    output_file_overwrite = False
    # numer of slices to analyze in the centre of each volume - must be even
    slice_range=20
    # factor by which to upsample image
    zoom=3
    # size in pixels of the rectangular segments to analyze a slice w/ ACF
    x_width = 40*zoom
    y_width = 10*zoom
    # multiplier to determine outlier threshold (mult*medianFWHM)
    mult=2
    #output options
    savePDF=True
    mkplots=True ## needs to be True for savePDF to work
    summaryvals=True
  
    os.chdir(scan_dir)

    if output_file_overwrite:
        if os.path.isfile(output_filename):
            os.remove(output_filename)
  
                      
    filename = f
    rawfile = f.split('_')[1:]
    name=f.split('.')[0]
    print 'Starting: '+name
    sessionID='_'.join(name.split('_')[1:])
    site=f.split('_')[1][0:3]
    f = scan_dir+f
                      
    afwhm = collections.defaultdict(list)
    bfwhm = collections.defaultdict(list)
    cfwhm = collections.defaultdict(list)
    dfwhm = collections.defaultdict(list)
    aacfs = collections.defaultdict(list)
    bacfs = collections.defaultdict(list)
    cacfs = collections.defaultdict(list)
    dacfs = collections.defaultdict(list)
              
    # getting raw and pre-processed scans
    image = nibabel.load(f)
    img = image.get_data()
    img = img.astype(np.float32, copy=False)
    rawf = rawscan_dir+site+"_FBN_NIFTI/"+'_'.join(rawfile)
    raw_image = nibabel.load(rawf)
    raw_img = raw_image.get_data()
    raw_img = raw_img.astype(np.float32, copy=False)
    
    roi_mean=0
    sc=0
    
    # getting dimensions of scan
    shapeslicex=np.shape(img)[0]
    shapeslicey=np.shape(img)[1]
    numslices=np.shape(img)[2]
    numTRs=np.shape(img)[3]
    
    ## check size of fMRI time series for validity
    if shapeslicex not in range(64,65) or shapeslicey not in range(64,65) or numslices not in range(35,45) or numTRs not in range(200,351):
        print name + ' - '
        print 'slice shape: (' + str(shapeslicex) + ',' + str(shapeslicey) + ')'
        print 'numslices: ' + str(numslices)
        print 'numTRs: ' + str(numTRs)
        return 0
    
    if summaryvals:
        fwhmacf = collections.defaultdict(list)
        
    ## Calculate global mean of raw scan
    for k in range(acqrem,numTRs-acqremend):
        ac=raw_img[:,:,:,k]
        for l in range(numslices/2 - slice_range/2, numslices/2 + slice_range/2 +1):
            sc+=1
            s = ac[:,:,l]
            roi_mean += np.mean(s) 
            
            ## ***find edges to locate centre of phantom in slice - needs testing***
            
#            #find centre of phantom in first slice
#            if (k==(numTRs-acqremend)/2) and (l == numslices/2):
#                snorm = s/np.max(np.abs(s))
#                edges=filter.sobel(snorm)
#                edges[edges<=0.1*np.max(edges)]=0
#                hough_radii = np.arange(15,30,1)
#                hough_res = hough_circle(edges,hough_radii)
#                centers=[]
#                accums = []
#                radii = []
#                for radius, h in zip(hough_radii,hough_res):
#                    peaks = peak_local_max(h,num_peaks=2)
#                    centers.extend(peaks)
#                    accums.extend(h[peaks[:,0],peaks[:,1]])
#                idx = np.argsort(accums)[::-1][:1]
#                center_y, center_x = centers[idx]
#                    print center_x
#                    print center_y 
    ## for now, centre is assumed to be stable at (32,32)       
    center_x=32*zoom
    center_y=32*zoom
    
    roi_mean = roi_mean/sc
    raw_img=[]
    ac=[]
    s=[]
            
    ## Calculate fwhm of all slices, and find mean
    a=[]
    b=[]
    c=[]
    d=[]
    
    maxa=collections.defaultdict(list)
    maxb=collections.defaultdict(list)
    maxc=collections.defaultdict(list)
    maxd=collections.defaultdict(list)
    meda=collections.defaultdict(list)
    medb=collections.defaultdict(list)
    medc=collections.defaultdict(list)
    medd=collections.defaultdict(list)
    mina=collections.defaultdict(list)
    minb=collections.defaultdict(list)
    minc=collections.defaultdict(list)
    mind=collections.defaultdict(list)
    ratiox=collections.defaultdict(list)
    ratioy=collections.defaultdict(list)
    
    
    for j in range(acqrem,numTRs-acqremend): 
        acquisition=img[:,:,:,j]
        afwhm[j]=collections.defaultdict(list)
        bfwhm[j]=collections.defaultdict(list)
        cfwhm[j]=collections.defaultdict(list)
        dfwhm[j]=collections.defaultdict(list)
        aacfs[j]=collections.defaultdict(list)
        bacfs[j]=collections.defaultdict(list)
        cacfs[j]=collections.defaultdict(list)
        dacfs[j]=collections.defaultdict(list)
        atemp=[]
        btemp=[]
        ctemp=[]
        dtemp=[]
        
        for i in range(numslices/2 - slice_range/2, numslices/2 + slice_range/2 +1): 
            pixdim = 0
            raw_slice = acquisition[:,:,i]                         
            image_mean = np.mean(raw_slice)
            sl=(raw_slice-image_mean)         
            
            ## upsample x2
            sl=scipy.ndimage.zoom(sl,zoom=zoom,order=5)
            
            sla=sl[center_y-y_width/2:center_y+y_width/2+1,center_x-x_width/2:center_x+x_width/2+1]
            slb = sla*0
            slc = sla*0
            sld = sla*0
            
            # create rotated rectangular segments of slice
            for indy in range(center_y-y_width/2,center_y+y_width/2+1):
                for indx in range(center_x-x_width/2,center_x+x_width/2+1):
                    x90,y90 = rotated_rectangle(indx,indy,xcen=center_x,ycen=center_y,theta=90)
                    slb[indy-center_y-y_width/2,indx-center_x-x_width/2]=sl[y90,x90]
                    x45,y45 = rotated_rectangle(indx,indy,xcen=center_x,ycen=center_y,theta=45)
                    slc[indy-center_y-y_width/2,indx-center_x-x_width/2]=sl[y45,x45]
                    xmin45,ymin45 = rotated_rectangle(indx,indy,xcen=center_x,ycen=center_y,theta=-45)
                    sld[indy-center_y-y_width/2,indx-center_x-x_width/2]=sl[ymin45,xmin45]
                    
            # remove mean and zero pad segments
            sla_padded=np.lib.pad(sla-np.mean(sla), ((y_width/2,y_width/2),(x_width/2,x_width/2)), 'constant',constant_values=(0,0))
            slb_padded=np.lib.pad(slb-np.mean(slb), ((y_width/2,y_width/2),(x_width/2,x_width/2)), 'constant',constant_values=(0,0))
            slc_padded=np.lib.pad(slc-np.mean(slc), ((y_width/2,y_width/2),(x_width/2,x_width/2)), 'constant',constant_values=(0,0))
            sld_padded=np.lib.pad(sld-np.mean(sld), ((y_width/2,y_width/2),(x_width/2,x_width/2)), 'constant',constant_values=(0,0))
            
            # adjust pixel dimensions by upsampling factor
            pixdim=image.get_header()['pixdim'][1]
            pixdimrs = pixdim/zoom

            # get ACF of segments
            acfra = acffunc(sla_padded)              
            acfrb = acffunc(slb_padded)
            acfrc = acffunc(slc_padded)
            acfrd = acffunc(sld_padded)

            fitaxis = np.linspace(0,x_width-1,x_width)
            
            # get centre cross-sections of ACFs
            acfa = acfra[y_width,x_width/2:x_width/2+x_width]
            acfb = acfrb[y_width,x_width/2:x_width/2+x_width]
            acfc = acfrc[y_width,x_width/2:x_width/2+x_width]
            acfd = acfrd[y_width,x_width/2:x_width/2+x_width]
            
            aacfs[j][i]=acfa
            bacfs[j][i]=acfb
            cacfs[j][i]=acfc
            dacfs[j][i]=acfd
            
            # get FWHM from ACF cross-sections, and convert from pixels to mm
            vala = fwhm(acfa)*pixdimrs
            valb = fwhm(acfb)*pixdimrs
            valc = fwhm(acfc)*pixdimrs
            vald = fwhm(acfd)*pixdimrs

            a.append(vala)
            b.append(valb)
            c.append(valc)
            d.append(vald)
            
            atemp.append(vala)
            btemp.append(valb)
            ctemp.append(valc)
            dtemp.append(vald)
            
            afwhm[j][i]=vala
            bfwhm[j][i]=valb
            cfwhm[j][i]=valc
            dfwhm[j][i]=vald
        
        # get summary values of volumes
        maxa[j]=np.max(atemp)
        meda[j]=np.median(atemp)
        mina[j]=np.min(atemp)
        maxb[j]=np.max(btemp)
        medb[j]=np.median(btemp)
        minb[j]=np.min(btemp)
        maxc[j]=np.max(ctemp)
        medc[j]=np.median(ctemp)
        minc[j]=np.min(ctemp)
        maxd[j]=np.max(dtemp)
        medd[j]=np.median(dtemp)
        mind[j]=np.min(dtemp)
        ratiox[j]=maxa[j]/minb[j]
        ratioy[j]=maxb[j]/mina[j]
              
    afwhmmean = np.median(a)
    bfwhmmean = np.median(b)
    cfwhmmean = np.median(c)
    dfwhmmean = np.median(d)
    if summaryvals:
        maxfwhmavar = np.var(maxa.values())
        maxfwhmbvar = np.var(maxb.values())
        maxfwhmcvar = np.var(maxc.values())
        maxfwhmdvar = np.var(maxd.values())
        minfwhmavar = np.var(mina.values())
        minfwhmbvar = np.var(minb.values())
        minfwhmcvar = np.var(minc.values())
        minfwhmdvar = np.var(mind.values())
        maxratiox = np.max(ratiox.values())
        maxratioy = np.max(ratioy.values())
	    
     
        fwhmacf['sessionID']=sessionID
        fwhmacf['amin']=np.min(a)
        fwhmacf['bmin']=np.min(b)
        fwhmacf['cmin']=np.min(c)
        fwhmacf['dmin']=np.min(d)
        fwhmacf['amax']=np.max(a)
        fwhmacf['bmax']=np.max(b)
        fwhmacf['cmax']=np.max(c)
        fwhmacf['dmax']=np.max(d)
        fwhmacf['amedian']=np.median(a)
        fwhmacf['bmedian']=np.median(b)
        fwhmacf['cmedian']=np.median(c)
        fwhmacf['dmedian']=np.median(d)
        fwhmacf['amean']=np.mean(a)
        fwhmacf['bmean']=np.mean(b)
        fwhmacf['cmean']=np.mean(c)
        fwhmacf['dmean']=np.mean(d)
        fwhmacf['amaxvar']=maxfwhmavar
        fwhmacf['bmaxvar']=maxfwhmbvar
        fwhmacf['cmaxvar']=maxfwhmcvar
        fwhmacf['dmaxvar']=maxfwhmdvar
        fwhmacf['aminvar']=minfwhmavar
        fwhmacf['bminvar']=minfwhmbvar
        fwhmacf['cminvar']=minfwhmcvar
        fwhmacf['dminvar']=minfwhmdvar
        fwhmacf['maxratiox_y']=maxratiox
        fwhmacf['maxratioy_x']=maxratioy 
        
        
    outlierimg = np.reshape(np.ravel(img[:,:,1,1]*0),(shapeslicex*shapeslicey,1))
    non_outlierimg = np.reshape(np.ravel(img[:,:,1,1]*0),(shapeslicex*shapeslicey,1))
    outlierpoints = np.ones((numslices, numTRs))
    outacfs = np.zeros(shapeslicex)
    inacfs = np.zeros(shapeslicex)
    #outlier count
    oc=0
    #non-outlier count
    noc=0
    
    if mkplots:
        plt.figure(figsize=(15,20),dpi=20)
        plt.subplot2grid((10,3),(0,0),colspan=2,rowspan=2)
    oa_fwhm=[]
    ob_fwhm=[]
    oc_fwhm=[]
    od_fwhm=[]
    na_fwhm=[]
    nb_fwhm=[]
    nc_fwhm=[]
    nd_fwhm=[]
    
    ## Find outliers based on multiplier*mean of fwhm
    for p in range(acqrem,numTRs-acqremend):
        for q in range(numslices/2 - slice_range/2, numslices/2 + slice_range/2 +1):   
            # if outlier                 
            if (afwhm[p][q] >= afwhmmean*mult) or (bfwhm[p][q] >= bfwhmmean*mult) or (cfwhm[p][q] >= cfwhmmean*mult) or (dfwhm[p][q] >= dfwhmmean*mult):
                flatimg = np.reshape(np.ravel((img[:,:,q,p]/roi_mean)*100),(shapeslicex*shapeslicey,1))
                outlierimg = np.concatenate((outlierimg,flatimg),1)
                outlierpoints[q][p] = -1
                oa_fwhm.append(afwhm[p][q])
                ob_fwhm.append(bfwhm[p][q])
                oc_fwhm.append(cfwhm[p][q])
                od_fwhm.append(bfwhm[p][q])
                if mkplots:
                    plt.plot(aacfs[p][q],c='r')   
                oc+=1
            # if non-outlier
            else:
                flatimg = np.reshape(np.ravel((img[:,:,q,p]/roi_mean)*100),(shapeslicex*shapeslicey,1))
                non_outlierimg = np.concatenate((non_outlierimg,flatimg),1)
                outlierpoints[q][p] = 1
                na_fwhm.append(afwhm[p][q])
                nb_fwhm.append(bfwhm[p][q])
                nc_fwhm.append(cfwhm[p][q])
                nd_fwhm.append(dfwhm[p][q])
                if mkplots:
                    plt.plot(aacfs[p][q],c='g')                            
                noc+=1     
            
    outlierimg=outlierimg[:,1:]
    non_outlierimg=non_outlierimg[:,1:]
    
    outlierimg=np.transpose(outlierimg)
    non_outlierimg=np.transpose(non_outlierimg)

## Output of results

    if mkplots:        
        plt.title(name+ ': ' + str(oc)+' Outlier Slices',fontsize=15)
        rnge=np.arange(acqrem,numTRs-acqremend)
        
        plt.subplot2grid((10,3),(2,0),colspan=3)
        plt.plot(rnge,maxa.values(),c='g',label='maxFWHM')
        plt.plot(rnge,meda.values(),c='r',label='medianFWHM') 
        plt.plot(rnge,mina.values(),c='b',label='minFWHM')
        plt.title('XFWHM - max,median,min',fontsize=15)
        plt.xlabel('Time (TR)',fontsize=15)
        plt.ylabel('FWHM (mm)',fontsize=15)
        plt.grid()
        plt.legend()
        
        
        plt.subplot2grid((10,3),(3,0),colspan=3)
        plt.plot(rnge,maxb.values(),c='g',label='maxFWHM')
        plt.plot(rnge,medb.values(),c='r',label='medianFWHM') 
        plt.plot(rnge,minb.values(),c='b',label='minFWHM') 
        plt.title('YFWHM - max,median,min',fontsize=15)
        plt.xlabel('Time (TR)',fontsize=15)
        plt.ylabel('FWHM (mm)',fontsize=15)
        plt.grid()
        plt.legend()
        
        plt.subplot2grid((10,3),(4,0),colspan=3)
        plt.plot(rnge,maxc.values(),c='g',label='maxFWHM')
        plt.plot(rnge,medc.values(),c='r',label='medianFWHM') 
        plt.plot(rnge,minc.values(),c='b',label='minFWHM') 
        plt.title('45deg FWHM - max,median,min',fontsize=15)
        plt.xlabel('Time (TR)',fontsize=15)
        plt.ylabel('FWHM (mm)',fontsize=15)
        plt.grid()
        plt.legend()
        
        plt.subplot2grid((10,3),(5,0),colspan=3)
        plt.plot(rnge,maxd.values(),c='g',label='maxFWHM')
        plt.plot(rnge,medd.values(),c='r',label='medianFWHM') 
        plt.plot(rnge,mind.values(),c='b',label='minFWHM') 
        plt.title('-45deg FWHM - max,median,min',fontsize=15)
        plt.xlabel('Time (TR)',fontsize=15)
        plt.ylabel('FWHM (mm)',fontsize=15)
        plt.grid()
        plt.legend()

        plt.subplot2grid((10,3),(6,0),colspan=3)
        outlierpointsimg = outlierpoints[numslices/2 - slice_range/2:numslices/2 + slice_range/2 +1,:]
        plt.imshow(outlierpointsimg, cmap="hot",interpolation='none',aspect='equal')
        plt.title('Outlier threshold [fwhmx,fwhmy] = [' + str(afwhmmean*mult) + ' , ' + str(bfwhmmean*mult) + ']',fontsize=15)                  
        plt.xlabel('Time (TR)',fontsize=15)
        plt.ylabel('Slice',fontsize=15)
        plt.tight_layout()
        
        plt.subplot2grid((10,3),(7,0),colspan=3)
        plt.plot(rnge,ratiox.values(),c='g',label='ratiox')
        plt.plot(rnge,ratioy.values(),c='r',label='ratioy') 
        plt.title('ratio max/min',fontsize=15)
        plt.xlabel('Time (TR)',fontsize=15)
        plt.ylabel('FWHM (mm)',fontsize=15)
        plt.grid()
        plt.legend()
        
        plt.subplot2grid((10,3),(8,0),rowspan=2)
        outim=np.reshape(np.mean(outlierimg,axis=0),np.shape(img[:,:,1,1]))
        plt.imshow(outim)
        plt.title('mean of outliers')
        plt.colorbar()

        plt.subplot2grid((10,3),(8,1),rowspan=2)
        noutim=np.reshape(np.mean(non_outlierimg,axis=0),np.shape(img[:,:,1,1]))
        plt.imshow(noutim)
        plt.title('mean of non-outliers')
        plt.colorbar()
        
        if savePDF:
            with PdfPages(scan_dir+name+analysis+suffix+'.pdf') as pdf:
                pdf.savefig()
            plt.close()
        else:
            plt.show()
            
    if summaryvals:
        fwhmacf['outlier_count']=oc
        print sessionID + ' has %d outliers' %oc
        fieldnames='sessionID,amin,bmin,cmin,dmin,amax,bmax,cmax,dmax,amedian,bmedian,cmedian,dmedian,amean,'\
        'bmean,cmean,dmean,amaxvar,bmaxvar,cmaxvar,dmaxvar,aminvar,bminvar,cminvar,dminvar,maxratiox_y,'\
        'maxratioy_x,outlier_count'.split(',')
        with open(output_filename,'ab+') as f:
		writer =csv.writer(f,delimiter=',')
		if os.stat(output_filename).st_size==0:
		    writer.writerow(fieldnames)
		row=[]
		for field in fieldnames:
		    row.append(fwhmacf[field])
		writer.writerow(row)
    print 'Done: '+name
       
    img=[]
    outlierimg=non_outlierimg=[]

if __name__ == '__main__':
    main(sys.argv[1])
