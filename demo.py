#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:27:36 2024

@author: fei.tan
"""

import numpy as np
import matplotlib.pyplot as plt
from digital_phantom import disk_phantom, resolution_phantom, low_contrast_phantom
from evaluation_metrics import geometric_accuracy, intensity_uniformity, percentage_ghosting, sharpness, snr_dual_image, high_contrast_resolution, threshold_determination, low_contrast_detectability

#%% Phantom Example
# parameters
fov = (240, 240) 
radius = (190/2 , 190/2)
edge = (100, 100)
center = (0, 0)
theta = 0
matrix_size = (128, 128)
intensity = 0.5
noise_std = 0.02

# disk example
kspace, disk = disk_phantom(fov, radius, center, theta, matrix_size, intensity, noise_std=noise_std)
plt.figure(figsize=(4,4), dpi=300)
plt.imshow(np.abs(disk), cmap='gray', vmax=1), plt.title('Disk Phantom'),  plt.axis('off')
plt.show()

# resolution phantom example
diameter_res = (4,4) # diameter represents resolution
array = (4,4)
radius_res = np.array(diameter_res) / 2
kspace_res, res = resolution_phantom(fov, radius_res, center=center, array=array, matrix_size=matrix_size, intensity=intensity, noise_std=noise_std)
plt.figure(figsize=(4,4), dpi=300)
plt.imshow(np.abs(res), cmap='gray',vmax=1), plt.axis('off')
plt.title('Resolution Phantom')
plt.show()

# low contrast phantom
radius_range = (0.75, 3.5)
nspokes = 10
disk_per_spoke = 3
contrast = 0.4 #0.2
kspace_lc, lc = low_contrast_phantom(fov, radius_range, center=center, nspokes=nspokes, spoke_dist=20, disk_per_spoke=disk_per_spoke, matrix_size=matrix_size, intensity=intensity, contrast=contrast, noise_std=noise_std)
plt.figure(figsize=(4,4), dpi=300)
plt.imshow(np.abs(lc), cmap='gray', vmax=1), plt.title('Low-Contrast Phantom'), plt.axis('off')
plt.show()

#%% Evaluation metric example 
# geometric accuracy
disk = np.abs(disk)
max_prc_err = geometric_accuracy(disk, fov, radius, center, plot=False)
print("geometric accuracy, maximum percentage error: ", max_prc_err)

# intenstiy uniformity
int_bias, int_var = intensity_uniformity(disk, fov, radius, center, intensity, plot=False, location_known=True)
print("intensity uniformity: ", int_var)

ghost_prc = percentage_ghosting(disk, fov, center, intensity, plot=False)
print('percentage ghosting: ', ghost_prc)

# sharpness
fwhm, sharp = sharpness(disk,  fov, radius, center, plot=False)
print('sharpness fwhm: ', fwhm, 'sharpness slope: ', sharp)

# snr
_, disk1 = disk_phantom(fov, radius, center, theta, matrix_size, intensity, noise_std=0.05)
_, disk2 = disk_phantom(fov, radius, center, theta, matrix_size, intensity, noise_std=0.05)
snr_dual = snr_dual_image(disk1, disk2, fov, radius, center, plot=False)
print('SNR dual: ', snr_dual)

# resolution
res = np.abs(res)
pixel_size, n_res_line = high_contrast_resolution(res, fov, radius=radius_res, center=center, array=(4,4), plot=False)
print('high contrast resolution, number of resolved line: ', n_res_line)

# low contrast detectability    
# threshold
for r in range(2,7):
    for noise in np.arange(0,0.05,0.01):
        thre, acc_max = threshold_determination(radius=r, noise_std=noise, patch_size=5, fov=fov, matrix_size=matrix_size, n_patch=300, contrast=0.1, plot=False)
# LCD
lc = np.abs(lc)
n_comp_sp, n_corr = low_contrast_detectability(lc, fov, radius_range, center=center, spoke_dist=20, intensity=intensity, contrast=contrast, noise_std=noise_std, plot=False)
print('low contrast detectability number of complete spokes: ', n_comp_sp)




