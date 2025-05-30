from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import cmath
import codecs
import chardet
from matplotlib.patches import Circle
import astropy.io.fits as fits
import warnings
warnings.filterwarnings('ignore')
import time

def multiply_mask(field_list):
    nx=ny=int(np.sqrt(len(field_list)))
    reshaped_array=np.rot90(np.reshape((field_list),(nx,ny)),axes=(-2,-1))
    mask = create_circular_mask(nx,ny,radius=nx/2)
    masked_img = reshaped_array.copy()
    masked_img[~mask] = 'nan'
    return masked_img

def no_mask(field_list):
    nx=ny=int(np.sqrt(len(field_list)))
    reshaped_array=np.rot90(np.reshape((field_list),(nx,ny)),axes=(-2,-1))
    reshaped_array[reshaped_array==0]='nan'
    masked_img = reshaped_array
    return masked_img


def plot_prm(prm_list,raytrace_list,mask='no'):
    
    x=np.array(raytrace_list)[:,0]
    y=np.array(raytrace_list)[:,1]
    l=len(prm_list)
    
    
    px=np.linspace(-1,1,int(np.sqrt(l)))
    py=np.linspace(-1,1,int(np.sqrt(l)))
    extent=[np.min(px),np.max(px),np.min(py),np.max(py)]
    
    fig1 = plt.figure(figsize=(10, 10))
    plots = []
    
    for i in range(3):
        for j in range(3):
            ax = plt.subplot2grid((3,3), (i,j))
            J_Mat=[]
            for n in range (l):
                j_val  =  prm_list[n][i][j]
                J_Mat.append(j_val)
            if mask=='yes' :
                J_array=multiply_mask(J_Mat)
            elif mask=='no':
                J_array=no_mask(J_Mat)
            im=ax.imshow(np.abs(J_array),cmap='magma',interpolation='nearest',extent=extent)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', width=0.5, labelsize=16)
            ax.set_ylabel('py', fontsize=14)
            ax.set_xlabel('px', fontsize=14)
            ax.set_title('A'+str(i+1)+str(j+1),fontdict={'fontsize': 20, 'fontweight': 'medium'})
            cb = fig1.colorbar(im,orientation='vertical',shrink=0.7)
            fig1.subplots_adjust(hspace=0.4, wspace=0.45, top=0.93, right=0.95)
            fig1.suptitle("PRM - Amplitude", fontsize=20)
    
    
    fig2 = plt.figure(figsize=(10, 10))
    plots = []
    for i in range(3):
        for j in range(3):
            ax2 = plt.subplot2grid((3,3), (i,j))
            J_Mat=[]
            for n in range (l):
                j_val  =  prm_list[n][i][j]
                J_Mat.append(j_val)
            if mask=='yes' :
                J_array=multiply_mask(J_Mat)
            elif mask=='no':
                J_array=no_mask(J_Mat)
            im=ax2.imshow(np.angle(J_array),cmap='coolwarm',interpolation='nearest',extent=extent)
            ax2.yaxis.set_ticks_position('both')
            ax2.xaxis.set_ticks_position('both')
            ax2.minorticks_on()
            ax2.tick_params(which='both', direction='in', width=0.5, labelsize=16)
            ax2.set_ylabel('py', fontsize=14)
            ax2.set_xlabel('px', fontsize=14)
            ax2.set_title('P'+str(i+1)+str(j+1),fontdict={'fontsize': 20, 'fontweight': 'medium'})
            cb = fig2.colorbar(im,orientation='vertical',shrink=0.7)
            fig2.subplots_adjust(hspace=0.4, wspace=0.45, top=0.93, right=0.95)
            fig2.suptitle("PRM - Phase", fontsize=20)
    
    return(fig1,fig2)

def plot_jones(jones_list,raytrace_list,mask='no'):
    
    x_M1=np.array(raytrace_list)[:,0]
    y_M1=np.array(raytrace_list)[:,1]
    
    l=len(prm_list)
    
    
    px=np.linspace(-1,1,int(np.sqrt(l)))
    py=np.linspace(-1,1,int(np.sqrt(l)))
    extent=[np.min(px),np.max(px),np.min(py),np.max(py)]

    amps=[['Axx','Axy'],['Ayx','Ayy']]
    pha=[['Phxx','Phxy'],['Phyx','Phyy']]

      
    fig1 = plt.figure(figsize=(8, 8))
    plots = []
    for i in range(2):
        for j in range(2):
            ax = plt.subplot2grid((2,2), (i,j))
            J_Mat=[]
            for n in range (l):
                j_val  =  jones_list[n][i][j]
                J_Mat.append(j_val)
            if mask=='yes' :
                J_array=multiply_mask(J_Mat)
            elif mask=='no':
                J_array=no_mask(J_Mat)
            im=ax.imshow(np.abs(J_array),cmap='magma',interpolation='nearest',extent=extent)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', width=0.5, labelsize=16)
            ax.set_ylabel('py', fontsize=14)
            ax.set_xlabel('px', fontsize=14)
            ax.set_title(str(amps[i][j]),fontdict={'fontsize': 20, 'fontweight': 'medium'})
            cb = fig1.colorbar(im,orientation='vertical',shrink=0.7)
            fig1.subplots_adjust(hspace=0.4, wspace=0.45, top=0.93, right=0.95)
            fig1.suptitle("PRM - Amplitude", fontsize=20)
    
    
    fig2 = plt.figure(figsize=(8, 8))
    plots = []
    for i in range(2):
        for j in range(2):
            ax2 = plt.subplot2grid((2,2), (i,j))
            J_Mat=[]
            for n in range (l):
                j_val  =  prm_list[n][i][j]
                J_Mat.append(j_val)
            if mask=='yes' :
                J_array=multiply_mask(J_Mat)
            elif mask=='no':
                J_array=no_mask(J_Mat)
            im=ax2.imshow(np.angle(J_array),cmap='coolwarm',interpolation='nearest',extent=extent)
            ax2.yaxis.set_ticks_position('both')
            ax2.xaxis.set_ticks_position('both')
            ax2.minorticks_on()
            ax2.tick_params(which='both', direction='in', width=0.5, labelsize=16)
            ax2.set_ylabel('py', fontsize=14)
            ax2.set_xlabel('px', fontsize=14)
            ax2.set_title(str(pha[i][j]),fontdict={'fontsize': 20, 'fontweight': 'medium'})
            cb = fig2.colorbar(im,orientation='vertical',shrink=0.7)
            fig2.subplots_adjust(hspace=0.4, wspace=0.45, top=0.93, right=0.95)
            fig2.suptitle("PRM - Phase", fontsize=20)
    
    return(fig1,fig2)

def save_jones(jones_list,raytrace_list,file_dir,fil,sur):

    nx=ny=np.sqrt(raytrace_list)

    comp=[['Exx','Exy'],['Eyx','Eyy']]
    l=len(jones_list)
    nx=ny=int(np.sqrt(l))
    for i in range(2):
        for j in range(2):
            J_Mat=[]
            for n in range (l):
                j_val  =  jones_list[n][i][j]
                J_Mat.append(j_val)
            
            np.array(J_Mat)[np.isnan(np.array(J_Mat))]=0
            J_Mat_real=np.real(J_Mat)
            E_real=np.rot90(np.reshape(J_Mat_real,(nx,ny)),axes=(-2,-1))
            hdu=fits.PrimaryHDU(E_real.astype(np.float64))
            hdul=fits.HDUList([hdu])
            hdul.writeto(file_dir+str(comp[i][j])+'_real_'+str(sur)+'_'+str(fil)+'.fits',overwrite=True)
    
            J_Mat_imag=np.imag(J_Mat)
            E_imag=np.rot90(np.reshape(J_Mat_imag,(nx,ny)),axes=(-2,-1))
            hdu=fits.PrimaryHDU(E_imag.astype(np.float64))
            hdul=fits.HDUList([hdu])
            hdul.writeto(file_dir+str(comp[i][j])+'_imag_'+str(sur)+'_'+str(fil)+'.fits',overwrite=True)
    return
        