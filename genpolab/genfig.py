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
    
    """
    Apply a circular mask to a 1D list of field values reshaped into a 2D array.

    Parameters:
    -----------
    field_list : array-like
        1D list or array of field values to be reshaped into a square 2D array.

    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (nx, ny) with a circular mask applied, where pixels outside
        the mask are set to NaN.

    Notes:
    ------
    - The input `field_list` is assumed to have a length that is a perfect square (nx * ny).
    - The function reshapes the input into a square 2D array and rotates it 90 degrees
      counterclockwise using `np.rot90` to align with the coordinate system.
    - A circular mask is created using `create_circular_mask` with a radius of nx/2,
      centered at the middle of the array.
    - Pixels outside the circular mask are set to NaN in the output array.
    - The input array is copied to avoid modifying the original data.
    """
    
    nx=ny=int(np.sqrt(len(field_list)))
    reshaped_array=np.rot90(np.reshape((field_list),(nx,ny)),axes=(-2,-1))
    mask = create_circular_mask(nx,ny,radius=nx/2)
    masked_img = reshaped_array.copy()
    masked_img[~mask] = 'nan'
    return masked_img

def no_mask(field_list):
    
    """
    Reshape a 1D list of field values into a 2D array and replace zeros with NaN.

    Parameters:
    -----------
    field_list : array-like
        1D list or array of field values to be reshaped into a square 2D array.

    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (nx, ny) where zero values are replaced with NaN.

    Notes:
    ------
    - The input `field_list` is assumed to have a length that is a perfect square (nx * ny).
    - The function reshapes the input into a square 2D array and rotates it 90 degrees
      counterclockwise using `np.rot90` to align with the coordinate system.
    - All zero values in the reshaped array are replaced with NaN.
    - No circular mask is applied, unlike related functions such as `multiply_mask`.
    """
    
    nx=ny=int(np.sqrt(len(field_list)))
    reshaped_array=np.rot90(np.reshape((field_list),(nx,ny)),axes=(-2,-1))
    reshaped_array[reshaped_array==0]='nan'
    masked_img = reshaped_array
    return masked_img


def plot_prm(prm_list,raytrace_list,mask='no'):

    """
    Plot amplitude and phase of Polarization Ray Tracing Matrices (PRMs) as 2D images.

    Parameters:
    -----------
    prm_list : list of numpy.ndarray
        List of 3x3 polarization rotation matrices for each ray.
    raytrace_list : list of tuples
        List of (x, y, z) coordinates for traced rays.
    mask : str, optional
        Type of mask to apply to the data ('yes' for circular mask, 'no' for no mask with zeros replaced by NaN).
        Defaults to 'no'.

    Returns:
    --------
    tuple
        A tuple containing:
        - fig1 : matplotlib.figure.Figure
            Figure containing 3x3 subplots showing the amplitude of PRM elements.
        - fig2 : matplotlib.figure.Figure
            Figure containing 3x3 subplots showing the phase of PRM elements.

    Notes:
    ------
    - The function assumes `prm_list` contains 3x3 matrices and `raytrace_list` contains coordinates
      for a number of rays equal to the square of an integer (for reshaping into a square grid).
    - The PRM elements are reshaped into a square 2D array (nx, ny) where nx = ny = sqrt(len(prm_list)).
    - The `mask` parameter determines whether a circular mask (`multiply_mask`) or no mask with zero-to-NaN
      replacement (`no_mask`) is applied to the PRM data.
    - Two figures are created:
      - `fig1` displays the amplitude (|PRM[i,j]|) using the 'magma' colormap.
      - `fig2` displays the phase (angle(PRM[i,j])) using the 'coolwarm' colormap.
    - Each subplot corresponds to a PRM element A_ij (amplitude) or P_ij (phase), with i,j = 1,2,3.
    - The plots use a normalized coordinate system (px, py) ranging from -1 to 1, derived from ray coordinates.
    """
    
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

    """
    Plot amplitude and phase of Jones matrices as 2D images.

    Parameters:
    -----------
    jones_list : list of numpy.ndarray
        List of 2x2 Jones matrices for each ray.
    raytrace_list : list of tuples
        List of (x, y, z) coordinates for traced rays.
    mask : str, optional
        Type of mask to apply to the data ('yes' for circular mask, 'no' for no mask with zeros replaced by NaN).
        Defaults to 'no'.

    Returns:
    --------
    tuple
        A tuple containing:
        - fig1 : matplotlib.figure.Figure
            Figure containing 2x2 subplots showing the amplitude of Jones matrix elements.
        - fig2 : matplotlib.figure.Figure
            Figure containing 2x2 subplots showing the phase of Jones matrix elements.

    Notes:
    ------
    - The function assumes `jones_list` contains 2x2 matrices and `raytrace_list` contains coordinates
      for a number of rays equal to the square of an integer (for reshaping into a square grid).
    - The Jones matrix elements are reshaped into a square 2D array (nx, ny) where nx = ny = sqrt(len(jones_list)).
    - The `mask` parameter determines whether a circular mask (`multiply_mask`) or no mask with zero-to-NaN
      replacement (`no_mask`) is applied to the Jones matrix data.
    - Two figures are created:
      - `fig1` displays the amplitude (|Jones[i,j]|) using the 'magma' colormap, labeled as Axx, Axy, Ayx, Ayy.
      - `fig2` displays the phase (angle(Jones[i,j])) using the 'coolwarm' colormap, labeled as Phxx, Phxy, Phyx, Phyy.
    - The plots use a normalized coordinate system (px, py) ranging from -1 to 1, derived from ray coordinates.
    - Note: The phase plot currently uses `prm_list` instead of `jones_list` for phase calculations, which may be a bug.
    - Colorbars are added to each subplot, and figures are adjusted for spacing and titles.
    """
    
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

    """
    Save real and imaginary parts of Jones matrix components as FITS files.

    Parameters:
    -----------
    jones_list : list of numpy.ndarray
        List of 2x2 Jones matrices for each ray.
    raytrace_list : list of tuples
        List of (x, y, z) coordinates for traced rays (used to determine grid size).
    file_dir : str
        Directory path where the FITS files will be saved.
    fil : str
        Identifier for the filter.
    sur : str
        Identifier for the surface or optical element.

    Returns:
    --------
    None
        Saves FITS files to the specified directory and returns nothing.

    Notes:
    ------
    - The function assumes `jones_list` contains 2x2 matrices and the number of rays
      (len(jones_list)) is a perfect square for reshaping into a square grid (nx, ny).
    - Each Jones matrix element (Exx, Exy, Eyx, Eyy) is processed separately, with real
      and imaginary parts saved as individual FITS files.
    - NaN values in the Jones matrix data are replaced with zeros before saving.
    - The data is reshaped into a square 2D array (nx, ny) and rotated 90 degrees
      counterclockwise using `np.rot90` to align with the coordinate system.
    - FITS files are named as `{comp}_real_{sur}_{fil}.fits` and `{comp}_imag_{sur}_{fil}.fits`,
      where `comp` is one of Exx, Exy, Eyx, Eyy.
    - The `raytrace_list` parameter is used to compute the grid size but is not directly
      used in the output; the code assumes `np.sqrt(len(raytrace_list))` is an integer.
    """

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
        