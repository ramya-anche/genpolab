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


def ref_coeff_two_layer(inc_ang,nb,ni,nf,df,wav):

    """
    Calculate reflection coefficients for s- and p-polarized light in a two-layer coating

    Parameters:
    -----------
    inc_ang : float
        Angle of incidence in radians.
    nb : float
        Refractive index of the metal layer.
    ni : float
        Refractive index of the incident medium (not used in calculations).
    nf : float
        Refractive index of the thin film protective layer.
    df : float
        Thickness of the film layer in microns.
    wav : float
        Wavelength of the incident light in microns.

    Returns:
    --------
    tuple
        A tuple containing the reflection coefficients for s-polarized (rs) and 
        p-polarized (rp) light.

    Notes:
    ------
    - The function assumes a two-layer system with a background medium, a film layer, 
      and an incident medium.
    - Calculations are based on Fresnel equations for reflection at interfaces, 
      accounting for phase changes due to the film thickness.
    - The incident medium refractive index (ni) is not used in the current implementation.
    """
    
    th_f=np.arcsin(np.sin(inc_ang)/nf)
    th_b=np.arcsin(nf*np.sin(th_f)/nb)
    eta_bs=nb*np.cos(th_b)
    eta_fs=nf*np.cos(th_f)
    eta_ms=1*np.cos(inc_ang)
    eta_bp=nb/np.cos(th_b)
    eta_fp=nf/np.cos(th_f)
    eta_mp=1/np.cos(inc_ang)
    del_f=(2*np.pi/float(wav))*nf*df*np.cos(th_f)
    Emp=np.cos(del_f)+1j*eta_bp*np.sin(del_f)/eta_fp
    Hmp=eta_bp*np.cos(del_f)+1j*eta_fp*np.sin(del_f)
    Ems=np.cos(del_f)+1j*eta_bs*np.sin(del_f)/eta_fs
    Hms=eta_bs*np.cos(del_f)+1j*eta_fs*np.sin(del_f) 

    rp=(eta_mp*Emp-Hmp)/(eta_mp*Emp+Hmp)
    rs=(eta_ms*Ems-Hms)/(eta_ms*Ems+Hms)


    return(rs,rp)

def ref_coeff_metal(inc_ang,nb,ni,nf=None,df=None,wav=None):
    
    """
    Calculate reflection coefficients for s- and p-polarized light at a single metal layer.

    Parameters:
    -----------
    inc_ang : float
        Angle of incidence in radians.
    nb : float
        Refractive index of the metal layer .
    ni : float
        Refractive index of the incident medium
    nf : float, optional
        Refractive index of the film layer (not used in calculations, defaults to None).
    df : float, optional
        Thickness of the film layer in microns (not used in calculations, defaults to None).
    wav : float, optional
        Wavelength of the incident light in microns (not used in calculations, defaults to None).

    Returns:
    --------
    tuple
        A tuple containing the reflection coefficients for s-polarized (rs) and 
        p-polarized (rp) light.

    Notes:
    ------
    - The function calculates reflection coefficients for a single layer of metal 
       Fresnel equations.
    - Parameters nf, df, and wav are included for compatibility with other functions 
      (e.g., ref_coeff_two_layer) but are not used in the calculations.
    - The angle of refraction is computed using Snell's law.
    """
    
    ang_refr=np.arcsin((np.sin(inc_ang)*ni)/nb)
    rs=(ni*np.cos(inc_ang)-nb*np.cos(ang_refr))/(ni*np.cos(inc_ang)+nb*np.cos(ang_refr))
    rp=(ni*np.cos(ang_refr)-nb*np.cos(inc_ang))/(ni*np.cos(ang_refr)+nb*np.cos(inc_ang))

    return(rs,rp)

# rotation matrix about an axis and angle
def rotation3D(angle,axis):
    
    """
    Generate a 3x3 rotation matrix for a 3D rotation around a specified axis by a given angle.

    Parameters:
    -----------
    angle : float
        Angle of rotation in radians.
    axis : array-like of shape (3,)
        Unit vector defining the axis of rotation (e.g., [x, y, z]).

    Returns:
    --------
    numpy.ndarray
        A 3x3 rotation matrix representing the rotation around the specified axis.

    Notes:
    ------
    - The rotation matrix is computed using the Rodrigues' rotation formula.
    """
    
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.array([[(1-c)*axis[0]**2 + c, (1-c)*axis[0]*axis[1] - s*axis[2], (1-c)*axis[0]*axis[2] + s*axis[1]],
                    [(1-c)*axis[1]*axis[0] + s*axis[2], (1-c)*axis[1]**2 + c, (1-c)*axis[1]*axis[2] - s*axis[0]],
                    [(1-c)*axis[2]*axis[0] - s*axis[1], (1-c)*axis[1]*axis[2] + s*axis[0], (1-c)*axis[2]**2 + c]])
    return mat

# computes the angle
def vectorAngle(u,v):
    
    """
    Calculate the angle between two vectors in 3D space.

    Parameters:
    -----------
    u : array-like of shape (3,)
        First input vector.
    v : array-like of shape (3,)
        Second input vector.

    Returns:
    --------
    float
        Angle between the two vectors in radians, in the range [0, π].

    Notes:
    ------
    - The vectors are normalized to unit length before computing the angle.
    - The function uses the dot product and handles cases where the angle is obtuse
      (dot product < 0) to ensure the correct angle is returned.
    """
    
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
    if u@v<0:
        return np.pi - 2*np.arcsin(np.linalg.norm(-v-u)/2)
    else:
        return 2*np.arcsin(np.linalg.norm(v-u)/2)


def cal_PRM(rp,rs,inc_dc,ref_dc,s_in,p_in,p_out):
    
    """
    Calculate the Polarization Ray tracing Matrix (PRM) for a given optical surface.

    Parameters:
    -----------
    rp : float
        Reflection coefficient for p-polarized.
    rs : float
        Reflection coefficient for s-polarized.
    inc_dc : array-like of shape (3,)
        Direction cosine of the incident ray.
    ref_dc : array-like of shape (3,)
        Direction cosine of the reflected ray.
    s_in : array-like of shape (3,)
        Direction cosine of the S-polarized component of the incident ray.
    p_in : array-like of shape (3,)
        Direction cosine of the P-polarized component of the incident ray.
    p_out : array-like of shape (3,)
        Direction cosine of the P-polarized component of the reflected ray.

    Returns:
    --------
    numpy.ndarray
        A 3x3 polarization rotation matrix (PRM) describing the transformation of
        polarization states from incident to reflected ray.

    Notes:
    ------
    - The function constructs a transformation matrix using the reflection coefficients
      and coordinate systems defined by the incident and reflected direction cosines
      and polarization vectors.
    - The input matrices `o_in` and `o_out` are built from the s-polarized, p-polarized,
      and direction cosine vectors.
    - The reflection matrix `ref_matrix` applies the reflection coefficients `rs` and `rp`
      for s- and p-polarizations, respectively
    - The PRM is computed as o_out @ ref_matrix @ inv(o_in), where inv(o_in) is the
      inverse of the incident coordinate matrix.
    """
    
    o_in=[[s_in[0], p_in[0], inc_dc[0]],
               [s_in[1], p_in[1], inc_dc[1]],
               [s_in[2], p_in[2], inc_dc[2]]]

    ref_matrix=[[rs,0,0],[0,rp,0],[0,0,1]]

    o_out=[[s_in[0], p_out[0], ref_dc[0]],
               [s_in[1], p_out[1], ref_dc[1]],
               [s_in[2], p_out[2], ref_dc[2]]]

    o_in_inv=np.linalg.inv(o_in)

    PRM=np.matmul(o_out,np.matmul(ref_matrix,o_in_inv))

    return(PRM)

def cal_loc_to_glob(inc_dc,ref_dc,a_loc,xin,xout):

    """
    Calculate transformation matrices to convert between local and global coordinate systems.

    Parameters:
    -----------
    inc_dc : array-like of shape (3,)
        Direction cosine of the incident ray.
    ref_dc : array-like of shape (3,)
        Direction cosine of the reflected ray.
    a_loc : array-like of shape (3,)
        Local axis vector defining the reference direction (e.g., surface normal).
    xin : array-like of shape (3,)
        Local x-axis vector for the incident coordinate system.
    xout : array-like of shape (3,)
        Local x-axis vector for the reflected coordinate system.

    Returns:
    --------
    tuple
        A tuple containing:
        - O_e : numpy.ndarray
            3x3 transformation matrix from local incident coordinates to global coordinates.
        - O_x_inv : numpy.ndarray
            3x3 inverse transformation matrix from global coordinates to local reflected coordinates.

    Notes:
    ------
    - The function uses a double pole coordinate system to define local coordinate systems
      for incident and reflected light relative to a local axis (a_loc).
    - The incident coordinate system is defined by `xin`, a derived y-axis (`yin`), and
      the incident direction cosine (`inc_dc`).
    - The reflected coordinate system is defined by `xout`, a derived y-axis (`yout`), and
      the reflected direction cosine (`ref_dc`).
    - Rotation matrices are computed using the `rotation3D` function to align local axes
      with the global coordinate system based on angles derived from the dot product and
      cross product of vectors.
    - Vectors are normalized to ensure unit length, and care is taken to handle numerical
      precision with `astype(np.float64)` for the y-axes.
    """
    
    # Double pole coordinate system
    kin = inc_dc
    r = np.cross(kin, a_loc)
    rin = r/np.linalg.norm(r) 
    thin = -np.arccos(np.dot(inc_dc,a_loc))
    Rin = rotation3D(thin, rin)
    xin=xin
    yin = np.cross(a_loc, xin)
    yin = yin.astype(np.float64)/np.linalg.norm(yin)  
    
    x = Rin @ xin
    x /= np.linalg.norm(x)
    y = Rin @ yin
    y /= np.linalg.norm(y)
    
    O_e = np.array([[x[0],y[0],inc_dc[0]],
                    [x[1],y[1],inc_dc[1]],
                    [x[2],y[2],inc_dc[2]]])


    r = np.cross(ref_dc,a_loc)
    rout = r/np.linalg.norm(r)
    th = -np.arccos(np.dot(ref_dc,a_loc))
    R = rotation3D(th,rout)

    yout = np.cross(a_loc,xout)
    yout = yout.astype(np.float64)/np.linalg.norm(yout)
    
    x = R @ xout
    x /= np.linalg.norm(x)
    y = R @ yout
    y /= np.linalg.norm(y)


    O_x = np.array([[x[0],y[0],ref_dc[0]],
                    [x[1],y[1],ref_dc[1]],
                    [x[2],y[2],ref_dc[2]]])


    O_x_inv=np.linalg.inv(O_x)
    return(O_e,O_x_inv)

def calc_jones(PRM_list,O_e_list,O_x_list):
    
    """
    Calculate Jones matrices for a sequence of optical transformations.

    Parameters:
    -----------
    PRM_list : list of numpy.ndarray
        List of 3x3 polarization ray tarcing (PRT) matrices  for each optical element.
    O_e_list : list of numpy.ndarray
        List of 3x3 transformation matrices from local incident coordinates to global coordinates.
    O_x_list : list of numpy.ndarray
        List of 3x3 transformation matrices from global coordinates to local reflected coordinates.

    Returns:
    --------
    list of numpy.ndarray
        A list of 3x3 Jones matrices in global coordinate system.

    Notes:
    ------
    - Each Jones matrix is computed as O_x @ PRM @ O_e, transforming the polarization state
      from the local incident coordinate system to the local reflected coordinate system via
      the global coordinate system.
    """
    
    jones_list=[]    
    for i in range(len(O_e_list)):
        jones=np.matmul(O_x_list[i],np.matmul(PRM_list[i],O_e_list[i]))
        jones_list.append(jones)
    return(jones_list)

# direction cosines for rays
def calc_prt(sur_list=1,n_rays=1,a_loc=1,xin=1,xout=1,n_layer=1,nb=1,ni=1,nf=None,df=None,wav=1):
    
    """
    Calculate polarization ray tracing matrix for a set of rays interacting with a surface.

    Parameters:
    -----------
    sur_list : Raytrace list at the surface which contains all the direction cosines.
    n_rays : int
        Number of rays traced.
    a_loc : array-like of shape (3,)
        Local axis vector defining the reference direction (e.g., surface normal).
    xin : array-like of shape (3,)
        Local x-axis vector for the incident ray coordinate system.
    xout : array-like of shape (3,)
        Local x-axis vector for the reflected ray coordinate system.
    n_layer : int
        Number of layers in the coating on the optical component (1 for  bare metal, 2 for protective layer).
    nb : float
        Refractive index of the bare metal layer.
    ni : float
        Refractive index of the incident medium.
    nf : float, optional
        Refractive index of the thin film layer (used for two-layer, defaults to None).
    df : float, optional
        Thickness of the film layer in microns (used for two-layer, defaults to None).
    wav : float
        Wavelength of the incident light in microns (defaults to 1).

    Returns:
    --------
    tuple
        A tuple containing:
        - vignetted_list : list of tuples
            List of coordinates for vignetted rays.
        - raytrace_list : list of tuples
            List of coordinates for traced rays.
        - PRM_list : list of numpy.ndarray
            List of 3x3 polarization ray tracing matrices (PRMs) for each ray.
        - O_e_list : list of numpy.ndarray
            List of 3x3 transformation matrices from local incident to global coordinates.
        - O_x_list : list of numpy.ndarray
            List of 3x3 transformation matrices from global to local reflected coordinates.

    Notes:
    ------
    - The function processes each ray to compute its interaction with a optical surface, including
      reflection coefficients, and coordinate transformations.
    - Vignetted rays are identified by the 'Ray-Stat' field in `sur_list` and are assigned
      zero values for direction cosines, angles, and polarization vectors.
    - Incident direction cosines are calculated from the reflected direction and surface normal.
    - Reflection coefficients (`rs`, `rp`) are computed using `ref_coeff_metal` for single-layer
      systems or `ref_coeff_two_layer` for two-layer systems.
    - Polarization vectors (`s_in`, `p_in`, `p_out`) are derived using cross products and normalized.
    - The function assumes `sur_list` contains keys: 'X-cor', 'Y-cor', 'Z-cor', 'Ray-Stat',
      'ref-x', 'ref-y', 'ref-z', 'nor-x', 'nor-y', 'nor-z', 'Inc-ang'.
    """

    
    PRM_list=[];O_e_list=[];O_x_list=[];vignetted_list=[];raytrace_list=[]
    for i in range(0,n_rays):
        sur=sur_list[i]
        coord=(float(sur['X-cor']),float(sur['Y-cor']),float(sur['Z-cor']))
        if sur['Ray-Stat']=='Vignetted':
            ref_dc=(0,0,0); nor_dc=(0,0,0); inc_dc=(0,0,0); inc_ang=0; rp=0; rs=0; s_in=(0,0,0); p_in=(0,0,0); p_out=(0,0,0);
            vignetted_list.append(coord)
        else: 
            
            raytrace_list.append(coord)
            ref_dc=(float(sur['ref-x']),float(sur['ref-y']),float(sur['ref-z']))
            nor_dc=(float(sur['nor-x']),float(sur['nor-y']),float(sur['nor-z']))
            inc_ang=np.radians(float(sur['Inc-ang']))
        
            in_ang=np.multiply(2,np.dot(ref_dc,nor_dc))
            inc_dc=ref_dc-np.multiply(in_ang,nor_dc)
            inc_dc=np.multiply(inc_dc,-1)
            inc_dc /=np.linalg.norm(inc_dc)
    
    
            ref_ang=np.arccos(np.dot(ref_dc,nor_dc))
            s_nor=np.cross(inc_dc,np.multiply(nor_dc,-1))
            s_in=s_nor/np.linalg.norm(s_nor)
            p_in=np.cross(inc_dc,s_in)
            p_in /=np.linalg.norm(p_in)
            p_out=np.cross(ref_dc,s_in)
            p_out /=np.linalg.norm(p_out)

            if n_layer==1 :
                rs=ref_coeff_metal(inc_ang,nb,ni,nf,df,wav)[0]
                rp=ref_coeff_metal(inc_ang,nb,ni,nf,df,wav)[1]
            elif n_layer==2:
                rs=ref_coeff_two_layer(inc_ang,nb,ni,nf,df,wav)[0]
                rp=ref_coeff_two_layer(inc_ang,nb,ni,nf,df,wav)[1]
    
        PRM=cal_PRM(rp,rs,inc_dc,ref_dc,s_in,p_in,p_out)
        
        O_e=cal_loc_to_glob(inc_dc,ref_dc,a_loc,xin,xout)[0]
        O_x=cal_loc_to_glob(inc_dc,ref_dc,a_loc,xin,xout)[1]
        
        PRM_list.append(PRM)
        O_e_list.append(O_e)
        O_x_list.append(O_x)
        
    return(vignetted_list,raytrace_list,PRM_list,O_e_list,O_x_list)

def create_circular_mask(h, w, center=None, radius=None):

    """
    Create a circular boolean mask for an image of specified height and width.

    Parameters:
    -----------
    h : int
        Height of the image (number of rows).
    w : int
        Width of the image (number of columns).
    center : tuple of int, optional
        Coordinates of the circle's center as (x, y). If None, defaults to the image center (w/2, h/2).
    radius : float, optional
        Radius of the circle. If None, defaults to the smallest distance from the center to the image edges.

    Returns:
    --------
    numpy.ndarray
        A boolean array of shape (h, w) where True indicates pixels inside the circle and False indicates
        pixels outside.

    Notes:
    ------
    - The mask is created using a grid of coordinates (via np.ogrid) to compute the Euclidean distance
      from each pixel to the center.
    - Pixels with a distance less than or equal to the radius are marked True in the mask.
    - The function assumes the image is represented as a 2D grid with (0,0) at the top-left corner.
    """

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

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