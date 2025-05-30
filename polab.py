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
    
    ang_refr=np.arcsin((np.sin(inc_ang)*ni)/nb)
    rs=(ni*np.cos(inc_ang)-nb*np.cos(ang_refr))/(ni*np.cos(inc_ang)+nb*np.cos(ang_refr))
    rp=(ni*np.cos(ang_refr)-nb*np.cos(inc_ang))/(ni*np.cos(ang_refr)+nb*np.cos(inc_ang))

    return(rs,rp)

# rotation matrix about an axis and angle
def rotation3D(angle,axis):
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.array([[(1-c)*axis[0]**2 + c, (1-c)*axis[0]*axis[1] - s*axis[2], (1-c)*axis[0]*axis[2] + s*axis[1]],
                    [(1-c)*axis[1]*axis[0] + s*axis[2], (1-c)*axis[1]**2 + c, (1-c)*axis[1]*axis[2] - s*axis[0]],
                    [(1-c)*axis[2]*axis[0] - s*axis[1], (1-c)*axis[1]*axis[2] + s*axis[0], (1-c)*axis[2]**2 + c]])
    return mat

# computes the angle
def vectorAngle(u,v):
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
    if u@v<0:
        return np.pi - 2*np.arcsin(np.linalg.norm(-v-u)/2)
    else:
        return 2*np.arcsin(np.linalg.norm(v-u)/2)


def cal_PRM(rp,rs,inc_dc,ref_dc,s_in,p_in,p_out):

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
    jones_list=[]    
    for i in range(len(O_e_list)):
        jones=np.matmul(O_x_list[i],np.matmul(PRM_list[i],O_e_list[i]))
        jones_list.append(jones)
    return(jones_list)

# direction cosines for rays
def calc_prt(sur_list=1,n_rays=1,a_loc=1,xin=1,xout=1,n_layer=1,nb=1,ni=1,nf=None,df=None,wav=1):
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

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

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