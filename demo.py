#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:27:36 2024

@authors: fei.tan, rongping.zeng
"""

import numpy as np
import matplotlib.pyplot as plt
from digital_phantom import disk_phantom, resolution_phantom, low_contrast_phantom
from evaluation_metrics import geometric_accuracy, intensity_uniformity, percentage_ghosting, sharpness, snr_dual_image, high_contrast_resolution, low_contrast_detectability
import os
import shutil

def main():
   
    #%% Phantom Example
    # parameters
    fov = (240, 240) 
    radius = (190/2 , 190/2)
    #edge = (100, 100)
    center = (0, 0)
    theta = 0
    matrix_size = (128, 128)
    intensity = 0.5 
    noise_std = 0.02

    path_output = os.path.abspath("./output/")
    if os.path.exists(path_output):
        shutil.rmtree(path_output) 
    os.mkdir(path_output)
    filename = os.path.join(path_output, 'result.txt')
    
    # disk example
    kspace, disk = disk_phantom(fov, radius, center, theta, matrix_size, intensity, noise_std=noise_std)
    plt.figure(1)
    plt.imshow(np.abs(disk), cmap='gray', vmax=1), plt.title('Disk Phantom'),  plt.axis('off')
    plt.savefig(os.path.join(path_output, 'Disk Phantom.png'))

    # resolution phantom example
    diameter_res = (4,4) # diameter represents resolution
    array = (4,4)
    radius_res = np.array(diameter_res) / 2
    kspace_res, res = resolution_phantom(fov, radius_res, center=center, array=array, matrix_size=matrix_size, intensity=intensity, noise_std=noise_std)
    plt.figure(2)
    plt.imshow(np.abs(res), cmap='gray',vmax=1), plt.title('Resolution Phantom'), plt.axis('off')
    plt.savefig(os.path.join(path_output, 'Resolution Phantom.png'))

    # low contrast phantom
    radius_range = (0.75, 3.5)
    nspokes = 10
    disk_per_spoke = 3
    contrast = 0.4 #0.2
    kspace_lc, lc = low_contrast_phantom(fov, radius_range, center=center, nspokes=nspokes, spoke_dist=20, disk_per_spoke=disk_per_spoke, matrix_size=matrix_size, intensity=intensity, contrast=contrast, noise_std=noise_std)
    plt.figure(3)
    plt.imshow(np.abs(lc), cmap='gray', vmax=1), plt.title('Low-Contrast Phantom'), plt.axis('off')
    plt.savefig(os.path.join(path_output, 'Low-Contrast Phantom.png'))

    #%% Evaluation metric example 
    # geometric accuracy
    disk = np.abs(disk)
    max_prc_err = geometric_accuracy(disk, fov, radius, center, plot=False)
    string ='Geometric accuracy, maximum percentage error: {:.4f}\n'.format(max_prc_err)
    print(string)
    with open(filename, "w") as file:
        file.write(string)

    # intenstiy uniformity
    int_bias, int_var = intensity_uniformity(disk, fov, radius, center, intensity, plot=False, location_known=True)
    string = 'Intensity uniformity: {:.4f}\n'.format(int_var)
    print(string)
    with open(filename, "a") as file:
        file.write(string)

    ghost_prc = percentage_ghosting(disk, fov, center, intensity, plot=False)
    string = 'Percentage ghosting: {:.4f}\n'.format(ghost_prc)
    print(string)
    with open(filename, "a") as file:
        file.write(string)

    # sharpness
    fwhm, sharp = sharpness(disk,  fov, radius, center, plot=False)
    string = 'Sharpness fwhm: {:.4f}; '.format(fwhm) + 'sharpness slope: {:.4f}\n'.format(sharp)
    print(string)
    with open(filename, "a") as file:
        file.write(string)

    # snr
    _, disk1 = disk_phantom(fov, radius, center, theta, matrix_size, intensity, noise_std=0.05)
    _, disk2 = disk_phantom(fov, radius, center, theta, matrix_size, intensity, noise_std=0.05)
    snr_dual = snr_dual_image(disk1, disk2, fov, radius, center, plot=False)
    string = 'SNR: {:.4f}\n'.format(snr_dual)
    print(string)
    with open(filename, "a") as file:
        file.write(string)

    # resolution
    res = np.abs(res)
    pixel_size, n_res_line = high_contrast_resolution(res, fov, radius=radius_res, center=center, array=(4,4), plot=False)
    string = 'High contrast resolution: \n \t number of resolved vertical lines in the upper resolution block: {:d} '.format(n_res_line[0]) + '\n \t number of resolved horizontal lines in the lower resolution block: {:d} \n'.format(n_res_line[1])    
    print(string)
    with open(filename, "a") as file:
        file.write(string)

    # low contrast detectability
    print('\nCalculating low contrast detectability measurement (may take a while) ...')
    lc = np.abs(lc)
    n_comp_sp, n_corr = low_contrast_detectability(lc, fov, radius_range, center=center, spoke_dist=20, intensity=intensity, contrast=contrast, noise_std=noise_std, plot=False)
    string = 'Low contrast detectability number of complete spokes: {:d}'.format(n_comp_sp)
    print(string)
    with open(filename, "a") as file:
        file.write(string)

    print('\n--------------------')
    print('Outputs are saved in', path_output)
 
    plt.show() #show plots at the end.

if __name__ == "__main__":
    main()
