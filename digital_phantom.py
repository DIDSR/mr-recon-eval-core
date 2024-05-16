#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:21:44 2024

@author: fei.tan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#%% basic phantom: circle & ellipse
def disk_phantom(fov, radius, center=(0,0), theta=0, matrix_size=(64,64), intensity=1, noise_std=0):
    '''
    Create disk phantom.
    
    Parameters
    ----------
    fov : tuple
        FOV in mm
    radius : tuple
        circle radius in mm
    center : tuple
        center in mm
    theta : float 
        rotation angle in degree
    matrix_size : tuple
        matrix size in number of pixel
    intensity : float
        intensity of the disk
    noise_std : float
        standard deviation of complex Gaussian noise

    Returns
    -------
    kspace : ndarray
        complex kspace of disk.
    disk : ndarray
        image.
 
    Reference
    ------
    https://github.com/philips-labs/Digital_Reference_Objects/blob/main/DRO.py
       
    '''
    
    # kspace grid points
    dk = 1 / np.asarray(fov)
    kx = dk[0] * (np.arange(matrix_size[0]) - matrix_size[0]//2)
    ky = dk[1] * (np.arange(matrix_size[1]) - matrix_size[1]//2)
    
    # kspace meshgrid
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    # rotation
    theta_radian = theta / 180 * np.pi # convert rad to degree
    Kx1 = Kx * np.cos(theta_radian) - Ky * np.sin(theta_radian)
    Ky1 = Kx * np.sin(theta_radian) + Ky * np.cos(theta_radian)
    
    # disk (radius[0]==radius[1]), ellipse (radius[0]!=radius[1])
    Kx1 = Kx1 * radius[0]
    Ky1 = Ky1 * radius[1]
    
    # create disk kspace, jinc function = Bessel(1,z)/z
    kr = np.sqrt(Kx1**2 + Ky1**2)
    z = kr * 2 * np.pi
    z[matrix_size[0]//2, matrix_size[1]//2] = 1e-10
    scale = intensity * (np.pi * radius[0] * radius[1]) / (fov[0]/matrix_size[0] * fov[1]/matrix_size[1])
    kspace =  2 * np.complex64(sp.special.jv(1, z) / z) * scale
    # scale by disk area and resolution, so that the disk intensity equals to 'intensity' 
    
    # center offset
    xphase = np.exp(-1j*(2*np.pi*center[0])*Kx)
    yphase = np.exp(-1j*(2*np.pi*center[1])*Ky)
    kspace = kspace * xphase * yphase
    
    # normalize
    kspace = kspace / np.sqrt(np.prod(kspace.shape))
    
    # add noise
    kspace += np.random.normal(0, noise_std, matrix_size) + 1j * np.random.normal(0, noise_std, matrix_size) 
    
    # ifft2 compute image
    disk = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace))) * np.sqrt(np.prod(kspace.shape))
    
    return kspace, disk

#%% compound phantoms: resolution and low-contrast
def resolution_phantom(fov, radius, center=(0,0), array=(4,4), matrix_size=(64,64), intensity=1, noise_std=0.1):
    '''
    Create resolution phantom.

    Parameters
    ----------
    fov : tuple
        Field of view in mm.
    radius : tuple
        Radii of individual holes in mm.
    center : tuple, optional
        Center of the phantom in mm. The default is (0,0).
    array : tuple, optional
        Number of holes in x,y dimension. The default is (4,4).
    matrix_size : tuple, optional
        Matrix size of kspace & image. The default is (64,64).
    intensity : float, optional
        Intensity of the image. The default is 1.
    noise_std : float, optional
        Standard deviation of additive complex gaussian noise. The default is 0.1.

    Returns
    -------
    kspace_res : ndarray
        Complex k-space.
    res : ndarray
        Image of resolution phantom.

    '''
    kspace_res = np.zeros(matrix_size, dtype=np.complex64)
    # upper left
    for y in range(array[1]):
        for x in range(array[0]):    
            kspace, disk = disk_phantom(fov, radius, np.array(center) - np.array(radius)*2 - np.array([x*4*radius[0] + y/2*radius[1], y*4*radius[1]]), theta=0, matrix_size=matrix_size, intensity=intensity)
            kspace_res += kspace

    # lower right
    for x in range(array[0]):
        for y in range(array[1]):
            # if x==0 and y==0:
            #     kspace_res = kspace_res
            # else:
            kspace, disk = disk_phantom(fov, radius, np.array(center) + np.array(radius)*2 + np.array([x*4*radius[0], y*4*radius[1]+x/2*radius[0]]), theta=0, matrix_size=matrix_size, intensity=intensity)
            kspace_res += kspace
                
    # add noise
    kspace_res = kspace_res + np.random.normal(0, noise_std, matrix_size) + 1j * np.random.normal(0, noise_std, matrix_size) 
    
    # compute resolution phantom
    res = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_res))) * np.sqrt(np.prod(kspace_res.shape))
    
    return kspace_res, res

def low_contrast_phantom(fov, radius_range, center=(0,0), nspokes=10, spoke_dist=20, disk_per_spoke=3, matrix_size=(64,64), intensity=1, contrast=0.1, noise_std=0.1):
    '''
    Generates low contrast phantom

    Parameters
    ----------
    fov : tuple
        FOV in mm.
    radius_range : tuple
        smallest and largest radii.
    center : tuple, optional
        center of the phantom in mm. The default is (0,0).
    nspokes : int, optional
        number of spokes. The default is 10.
    spoke_dist : float, optional
        distance between phantoms within one spoke. The default is 20.
    disk_per_spoke : int, optional
        number of disks per spoke. The default is 3.
    matrix_size : tuple, optional
        matrix size. The default is (64,64).
    intensity : float, optional
        intensity of background phantom. The default is 1.
    contrast : float, optional
        additive intensity of the foreground phantoms. The default is 0.1.
    noise_std : float, optional
        noise standard deviation. The default is 0.1.

    Returns
    -------
    kspace_lc : ndarray
        kspace of low contrast phantom.
    lc : ndarray
        image of low contrast phantom.

    '''
    # create background disk
    kspace_lc, _ = disk_phantom(fov, (spoke_dist*(disk_per_spoke+2), spoke_dist*(disk_per_spoke+2)), center=center, theta=0, matrix_size=matrix_size, intensity=intensity)

    # compute radii and rotation angle, large to small
    radius_list = np.linspace(radius_range[1], radius_range[0], num=nspokes)
    theta = 2 * np.pi / nspokes
    
    # create small low contrast disks
    for spoke in range(nspokes):
        for ind in range(disk_per_spoke):
            x = np.sin(theta * spoke) * spoke_dist * (ind + 2) + center[0]
            y = np.cos(theta * spoke) * spoke_dist * (ind + 2) + center[1]
            kspace, disk = disk_phantom(fov, (radius_list[spoke], radius_list[spoke]), (x,y), 0, matrix_size, intensity*contrast)
            kspace_lc += kspace
    
    # add noise
    kspace_lc = kspace_lc + np.random.normal(0, noise_std, matrix_size) + 1j * np.random.normal(0, noise_std, matrix_size) 
 
    # compute resolution phantom
    lc = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_lc))) * np.sqrt(np.prod(kspace_lc.shape))
    
    return kspace_lc, lc
 