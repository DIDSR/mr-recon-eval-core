#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:23:25 2024

@author: fei.tan
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import scipy as sp
from digital_phantom import disk_phantom

#%% 1. Geometric Accuracy
def geometric_accuracy(disk, fov, radius, center, plot=False):
    '''
    Measure the geometric accuracy of a disk phantom.
    
    Parameters
    ----------
    disk : (M, N) ndarray
        Image to measure.
    fov : tuple
        Field of view in mm (fov_x, fov_y).
    radius : tuple
        Ground truth radius in mm.
    center : tuple
        Ground truth center in mm.
    plot : bool, optional
        Plot predicted axes, center, and bounding box. The default is False.

    Returns
    -------
    max_percentage_error : float
        Maximum percentage error of major axis and minor axis compared with ground truth.
    eccentricity : float
        Equals to 0 when input image is circle, range [0, 1).

    References
    -------
    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html#sphx-glr-auto-examples-segmentation-plot-regionprops-py
    '''
    # get image resolution in mm
    matrix_size = np.shape(disk)
    resolution = np.array(fov) / matrix_size
    
    # histogram thresholding
    blurred_image = ski.filters.gaussian(disk, sigma=1.0) # blur the image to denoise    
    t = ski.filters.threshold_otsu(blurred_image) # perform automatic thresholding
    
    # measure properties
    disk_label = np.int8(disk > t)
    disk_label = ski.morphology.remove_small_objects(disk_label > 0, min_size=128)
    disk_label = ski.morphology.remove_small_holes(disk_label > 0)
    disk_label = ski.measure.label(disk_label)
    props = ski.measure.regionprops_table(disk_label, disk, spacing = resolution, properties=['centroid','axis_minor_length','axis_major_length','eccentricity'])
    
    # find the center closest to ground truth
    center_dist = np.sqrt((props['centroid-0'] - center[0])**2 + (props['centroid-1'] - center[1])**2)
    ind = np.argmin(center_dist)
    
    # calculate percentage error
    max_percentage_error = max(abs(props['axis_major_length'][ind]/2 - max(radius)) / max(radius), abs(props['axis_minor_length'][ind]/2 - min(radius)) / min(radius))
    eccentricity = props['eccentricity']
    
    # plot prediction results (visualize & debug)
    if plot == True:
        props = ski.measure.regionprops(disk_label, disk, spacing = resolution)
        fig, ax = plt.subplots(figsize=(4,4), dpi=300)
        ax.imshow(disk, cmap=plt.cm.gray)
        plt.axis("off")
        for prop in props:
            # short axis, long axis, and center
            y0, x0 = np.array(prop.centroid) / resolution
            orientation = prop.orientation
            x1 = x0 + np.cos(orientation) * 0.5 * prop.axis_minor_length / resolution[1]
            y1 = y0 - np.sin(orientation) * 0.5 * prop.axis_minor_length / resolution[0]
            x2 = x0 - np.sin(orientation) * 0.5 * prop.axis_major_length / resolution[1]
            y2 = y0 - np.cos(orientation) * 0.5 * prop.axis_major_length / resolution[0]      
            ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            # ax.plot(x0,y0, '.g', markersize=15)
            
            # bounding box
            minr, minc, maxr, maxc = prop.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            # ax.plot(bx, by, '--w', linewidth=2.5)
        plt.show()
        
    return max_percentage_error

#%% 2. Intensity uniformity 
def intensity_uniformity(disk, fov, radius, center, intensity, plot=False, location_known=True):
    '''
    Measure the intensity uniformity using a disk phantom.
    
    Parameters
    ----------
    disk : (M, N) ndarray
        Image to measure.
    fov : tuple
        Field of view in mm (fov_x, fov_y).
    radius : tuple
        Ground truth radius in mm.
    center : tuple
        Ground truth center in mm.
    plot : bool, optional
        Plot predicted axes, center, and bounding box. The default is False.
    location_known : bool, optional
        Consider the center of the disk known or known. The default is True. 
        If known, use the known center to define ROI. If unknown, use thresholding to detect ROI.  

    Returns
    -------
    intensity_bias : float
        Intenstiy bias, percentage error of mean intensity compared with ground truth.
    intensity_uniform : float
        Intensity uniformity
        
    '''
    # get image resolution in mm
    matrix_size = np.asarray(np.shape(disk)) 
    pixel_size = np.asarray(fov) / matrix_size
    disk = np.abs(disk)
    disk_ori = disk
    
    if location_known:
        # define mask with center and radius 
        center = np.asarray(center)
        radius = np.asarray(radius)
        
        center_pixel = np.int8(np.round(center / pixel_size + matrix_size / 2))
        radius_pixel = np.int8(np.round(radius / pixel_size))
       
        mask = np.zeros(matrix_size, dtype=np.uint8)
        rr, cc = ski.draw.disk(center_pixel, radius_pixel[0], shape=matrix_size)
        mask[rr, cc] = 1
        
    else: 
        # define mask with thresholding
        # histogram thresholding
        blurred_image = ski.filters.gaussian(disk, sigma=1.0) # blur the image to denoise    
        t = ski.filters.threshold_otsu(blurred_image) # perform automatic thresholding
        
        # morphology
        mask = disk > t
        mask = ski.morphology.remove_small_objects(mask, min_size=128)
        mask = ski.morphology.remove_small_holes(mask)
        
    disk_erode = ski.morphology.binary_erosion(mask, ski.morphology.disk(5.0))
    
    # NEMA standard low-pass filter
    kernel = np.array([[1,2,1], [2,4,2], [1,2,1]])
    kernel = kernel / np.sum(kernel)
    disk = sp.signal.convolve2d(disk, kernel, mode='same')
    
    # intensity max, min, mean
    intensity_max = np.amax(disk[disk_erode])
    intensity_min = np.amin(disk[disk_erode])
    intensity_mean = np.mean(disk[disk_erode])
    
    # bias, variation
    intensity_bias = (intensity_mean - intensity) / intensity
    intensity_uniform = 100 * (1 - (intensity_max - intensity_min) / (intensity_max + intensity_min))
    
    if plot:
            
        plt.figure(figsize=(4,4)) 
        plt.imshow(disk_ori, cmap='gray')
        plt.imshow(disk_erode, cmap = 'autumn', alpha=disk_erode*0.8)
        
        max_loc = np.argwhere(disk == intensity_max)
        min_loc = np.argwhere(disk == intensity_min)
        plt.plot(max_loc[:,0], max_loc[:,1], 'rP')
        plt.plot(min_loc[:,0], min_loc[:,1], 'bX')
        plt.axis('off')
        plt.show()

    return intensity_bias, intensity_uniform

#%% 3. percentage ghosting
def percentage_ghosting(disk, fov, center, intensity, plot=False):
    '''
    Measure the percentage ghosting of a disk phantom.
    
    Parameters
    ----------
    disk : (M, N) ndarray
        Image to measure.
    fov : tuple
        Field of view in mm (fov_x, fov_y).
    radius : tuple
        Ground truth radius in mm.
    center : tuple
        Ground truth center in mm.
    plot : bool, optional
        Plot predicted axes, center, and bounding box. The default is False.

    Returns
    -------
    ghosting_ratio : float
        Ghosting ratio, percentage image ghosting

    '''
    # get image resolution in mm
    matrix_size = np.shape(disk)
    pixel_size = np.array(fov) / matrix_size
    disk = np.abs(disk)
    
    # histogram thresholding
    blurred_image = ski.filters.gaussian(disk, sigma=1.0) # blur the image to denoise    
    t = ski.filters.threshold_otsu(blurred_image) # perform automatic thresholding
    
    # measure properties
    disk_label = np.int8(disk > t)
    disk_erode = ski.morphology.binary_erosion(disk_label, ski.morphology.disk(5.0))
    
    # intensity mean
    intensity_mean = np.mean(disk[disk_erode])
    
    # define background boxes
    prc_dist_short_edge = 32
    prc_dist_long_edge = 4
    
    top_x = slice(np.int8(matrix_size[0]/prc_dist_long_edge), np.int8(3*matrix_size[0]/prc_dist_long_edge))
    top_y = slice(np.int8(matrix_size[1]/prc_dist_short_edge), np.int8(3*matrix_size[1]/prc_dist_short_edge))
    
    bottom_x = slice(np.int8(matrix_size[0]/prc_dist_long_edge), np.int8(3*matrix_size[0]/prc_dist_long_edge))
    bottom_y = slice(np.int8(matrix_size[1]*(1 - 3/prc_dist_short_edge)), np.int8(matrix_size[1] * (1 - 1/prc_dist_short_edge)))
    
    left_x = slice(np.int8(matrix_size[0]/prc_dist_short_edge), np.int8(3*matrix_size[0]/prc_dist_short_edge))
    left_y = slice(np.int8(matrix_size[1]/prc_dist_long_edge), np.int8(3*matrix_size[1]/prc_dist_long_edge))
    
    right_x = slice(np.int8(matrix_size[0] * (1 - 3/prc_dist_short_edge)), np.int8(matrix_size[0] * (1 - 1/prc_dist_short_edge)))
    right_y = slice(np.int8(matrix_size[1]/prc_dist_long_edge), np.int8(3*matrix_size[1]/prc_dist_long_edge))
    
    top = disk[top_x, top_y]
    bottom = disk[bottom_x, bottom_y]
    left = disk[left_x, left_y]
    right = disk[right_x, right_y]
    
    ghosting_ratio = np.abs((np.mean(top) + np.mean(bottom)) - (np.mean(left) + np.mean(right))) / (2 * intensity_mean)

    if plot:
        plt.figure(figsize=(4,4), dpi=300)
        mask = np.zeros(matrix_size)
        mask[top_x,top_y]=1
        mask[bottom_x, bottom_y] = 1
        mask[left_x, left_y] = 1
        mask[right_x, right_y] = 1
        
        plt.imshow(disk, cmap = 'gray')
        plt.imshow(mask, cmap = 'autumn', alpha= mask*0.9)
        plt.axis('off')
        plt.show()
        
    return ghosting_ratio

#%% 4. sharpness for disk
def sharpness(disk, fov, radius, center, plot=False, fit=False):
    '''
    Sharpness using the edge spread function of a disk phantom.

    Parameters
    ----------
    disk : ndarray
        disk image.
    fov : tuple
        FOV of the disk.
    radius : tuple
        radius of the disk.
    center : tuple
        center of the disk.
    plot : boolean, optional
        Plot figures for debugging. The default is False.
    fit : boolean, optional
        Whether to fit the edge spread function to sigmoid. The default is False.

    Returns
    -------
    fwhm : float
        Full-width-half-maximum of the fitted Lorentzian function 
    mal_val : float
        Maximum value of the fitted Lorentzian function
    
    References
    -------
    https://doi.org/10.1016/j.neuroimage.2020.117227
    https://doi.org/10.1118/1.4725171
    https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
    
    '''
    matrix_size = np.asarray(np.shape(disk))
    pixel_size = np.array(fov) / np.array(matrix_size)
    center = np.asarray(center)
    radius = np.asarray(radius)
    
    # image space grid points
    x = pixel_size[0] * (np.arange(matrix_size[0]) - matrix_size[0]//2) - center[0]
    y = pixel_size[1] * (np.arange(matrix_size[1]) - matrix_size[1]//2) - center[1]
    
    # kspace meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    max_dist = np.amax(radius) * 2
    
    # distance to center and intensity
    distance_to_center = np.sqrt(X**2 + Y**2)
    
    distance = distance_to_center.flatten()
    dist = distance
    distance = distance[dist < (max_dist+1)]
    distance = distance - np.arange(len(distance)) * 1e-12 # resolve duplicates
    
    intensity = np.abs(disk.flatten())
    intensity = intensity[dist < (max_dist+1)]
    
    # re-bin into uniform distance
    dr = np.amin(pixel_size) / 10
    distance_norm = np.arange(0, np.amax(radius)*2, dr)
    max_ind = distance_norm < np.amax(distance)
    distance_norm = distance_norm[max_ind]
    
    # ESF
    if not fit:    # 1st order spline interpolate
        f = sp.interpolate.interp1d(distance, intensity, kind='slinear')
        intensity_norm = f(distance_norm)
    else:       # fit to a sigmoid
        # define sigmoid function
        def sigmoid(x, x0, k, a, b):
            y = a / (1 + np.exp(-k * (x - x0))) + b
            return y
        # initialization
        p0 = [np.median(distance), 1, np.amax(intensity) - np.amin(intensity), np.amin(intensity)]
        # curve fitting
        popt, pcov = sp.optimize.curve_fit(sigmoid, distance, intensity, p0, method='lm')
        # print(popt)
        intensity_norm = sigmoid(distance_norm, popt[0], popt[1], popt[2], popt[3])
        
    # LSF
    lsf = np.zeros_like(intensity_norm)
    lsf[1:-1] = np.abs(intensity_norm[2:] - intensity_norm[:-2]) / dr / 2
    hann = np.hanning(len(max_ind))
    hann = hann[max_ind]
    lsf = lsf * hann
    
    # Fit to lorenzian function
    def lorentzian(x, gamma, x0, a, b):    
        return a * 0.25 * gamma**2 / ((x - x0)**2 + 0.25 * gamma**2) + b
    
    p1 = [0.01, np.amax(radius), 0.1, 0]
    popt1, pcov1 = sp.optimize.curve_fit(lorentzian, distance_norm, lsf, p1)
    lsf_fitted = lorentzian(distance_norm, popt1[0], popt1[1], popt1[2], popt1[3])
    
    # FWHM
    fwhm = popt1[0] # FWHM lorentzian
    max_val = popt1[2] + popt1[3] # max value lorentzian
    half_max_x1 = popt1[1] - popt1[0]/2
    half_max_x2 = popt1[1] + popt1[0]/2
    half_max_val = popt1[2]/2 + popt1[3]
    
    # MTF
    mtf = np.abs(np.fft.fftshift(np.fft.fft(lsf))) / np.sum(lsf)
    kmax = 1 / dr
    k = np.linspace(-kmax/2, kmax/2, num=len(mtf), endpoint=False)
    
    if plot:
        plt.subplots(figsize=(7,6), dpi=600)
        plt.subplot(2,1,1)
        plt.plot(distance, intensity, ".")
        plt.plot(distance_norm, intensity_norm)
        # plt.xlabel("distance from center [mm]")
        plt.ylabel("normalized \n pixel intensity")
        plt.legend(["data","rebinned"])
        plt.xlim([80, 110])
        
        plt.subplot(2,1,2)
        plt.plot(distance_norm, lsf, '.')
        plt.plot(distance_norm, lsf_fitted)

        # FWHM
        plt.plot([half_max_x1, half_max_x2],[half_max_val, half_max_val], 'r--')
        plt.legend(["data", "fitted", "FWHM"])
        plt.xlim([80, 110])
        plt.xlabel("distance from center [mm]")
        plt.ylabel("derivative of \n pixel intensity")
        plt.show()
    
    return fwhm, max_val

## 5. SNR measurement following NEMA requirement
def snr_dual_image(disk1, disk2, fov, radius, center, plot=False, signal_roi_width=7, noise_roi_width=13):
    '''
    SNR measurement using two disks.

    Parameters
    ----------
    disk1 : ndarray
        1st disk phantom.
    disk2 : ndarray
        2nd disk phantom.
    fov : tuple
        FOV of the disk phantoms in mm.
    radius : tuple
        Radius of the disk phantoms in mm.
    center : tuple
        Center of the disk phantoms in mm.
    plot : Boolean, optional
        Plot for debugging. The default is False.
    signal_roi_width : int, optional
        Signal ROI width. The default is 7.
    noise_roi_width : TYPE, optional
        Noise ROI width. noise_roi_width >= 11, The default is 13.

    Returns
    -------
    snr : float
        SNR.

    '''
    
    # get image resolution in mm
    matrix_size = np.asarray(np.shape(disk1)) 
    pixel_size = np.asarray(fov) / matrix_size
    disk1 = np.abs(disk1)
    disk2 = np.abs(disk2)
    disk3 = disk1 - disk2
    
    # define mask with center and radius 
    center = np.asarray(center)
    radius = np.asarray(radius)
    
    center_pixel = np.int8(np.round(center / pixel_size + matrix_size / 2))
    radius_pixel = np.int8(np.round(radius / pixel_size))
    
    if np.amin(radius_pixel) < noise_roi_width * np.sqrt(2): # noise ROI at lease 11x11 pixel
        print("Disk smaller than required ROI, please increase disk radius")
        return
    
    # define signal and noise ROI
    signal_roi = np.zeros(matrix_size, dtype=np.uint8)
    signal_roi[center_pixel[0] - signal_roi_width // 2 : center_pixel[0] + signal_roi_width // 2 + 1, center_pixel[1] - signal_roi_width // 2 : center_pixel[1] + signal_roi_width // 2 + 1] = 1

    noise_roi = np.zeros(matrix_size, dtype=np.uint8)
    noise_roi[center_pixel[0] - noise_roi_width // 2 : center_pixel[0] + noise_roi_width // 2 + 1, center_pixel[1] - noise_roi_width // 2 : center_pixel[1] + noise_roi_width // 2 + 1] = 1
    
    # reject pixels with value lower than 5 std
    p = np.std(disk3[signal_roi>0])
    accepted_pixel = disk1 > 5 * p
    noise_roi_accepted = noise_roi * accepted_pixel
    
    if np.sum(noise_roi_accepted) < 121:
        print("not enough pixels in noise ROI, please increase noise roi width")
    
    # calculate SNR
    s = np.mean(disk1[signal_roi>0])
    noise_std = np.std(disk3[noise_roi_accepted>0])
    snr = s * np.sqrt(2) / noise_std
    
    if plot:
        plt.figure(figsize=(12,4), dpi=300)
        plt.subplot(1,3,1), plt.imshow(disk1, cmap='gray'), plt.imshow(signal_roi, cmap='Blues', alpha=signal_roi*0.9)
        plt.title('Disk 1'), plt.axis('off')
        plt.subplot(1,3,2), plt.imshow(disk2, cmap='gray')
        plt.title('Disk 2'), plt.axis('off')
        plt.subplot(1,3,3), plt.imshow(disk3, cmap='gray'), plt.imshow(noise_roi_accepted, cmap='autumn', alpha=noise_roi_accepted*0.9)
        plt.title('Difference'), plt.axis('off')
        
        plt.show()
        
    return snr
    
#%% 6. High contrast resolution
def high_contrast_resolution(res, fov, radius, center=(0,0), array=(4,4), plot=True):
    '''
    High contrast resolution measurement using resolution phantom.

    Parameters
    ----------
    res : ndarray
        Image of resolution phantom.
    fov : tuple
        Field of view in mm.
    radius : tuple
        Radii of individual holes in mm.
    center : tuple, optional
        Center of the phantom in mm. The default is (0,0).
    array : tuple, optional
        Number of holes in x,y dimension. The default is (4,4).
    plot : boolean, optional
        Plot for debugging. The default is True.

    Returns
    -------
    pixel_size : tuple
        pixel size in both directions.
    number_of_resolved_line : tuple
        number of resolved lines in both directions.

    '''
    # initialization
    matrix_size = np.shape(res)
    pixel_size = np.array(fov) / np.array(matrix_size)
    n_res_v = np.zeros((array[1],1))
    n_res_h = np.zeros((array[0],1))
    
    # line profile in upper left
    for y in range(array[1]):
        # line position
        loc = np.int8(np.round((center[1] - 2*radius[1] - y*4*radius[1]) / pixel_size[1] + matrix_size[1] / 2)) # location of the line in # pixels
        line = res[:, loc]
        
        # find peaks
        dist = max(1, radius[1] * 2 / pixel_size[1]) # minimum distance between peaks
        t = ski.filters.threshold_otsu(line) # histogram thresholding, minimum height of peaks
        peaks, _ = sp.signal.find_peaks(line, distance=dist, height=t) # find peaks
        
        # number of peaks == array
        n_res_v[y] = len(peaks) == array[0]
        
        if plot:
            # plot for debugging
            plt.figure(figsize=(4,4), dpi=600)
            plt.rc('font', size=12) 
            plt.plot(np.arange(matrix_size[1])*pixel_size[1], line)
            plt.plot(peaks*pixel_size[1], line[peaks], "x")
            plt.xlabel('location [mm]')
            plt.ylabel('intensity')
            plt.show()
        
    # line profile in lower right
    for x in range(array[0]):
       # line position
       loc = np.int8(np.round((center[0] + 2*radius[0] + x*4*radius[0]) / pixel_size[0] + matrix_size[0] / 2)) # location of the line in # pixels
       line = res[loc, :]
       
       # find peaks
       dist = max(1, radius[0] * 2 / pixel_size[0]) # minimum distance between peaks
       t = ski.filters.threshold_otsu(line) # histogram thresholding, minimum height of peaks
       peaks, _ = sp.signal.find_peaks(line, distance=dist, height=t) # find peaks
       
       # number of peaks == array
       n_res_h[x] = len(peaks) == array[1]
       
       if plot:
           # plot for debugging
           plt.figure(figsize=(4,4), dpi=600)
           plt.rc('font', size=12) 
           plt.plot(np.arange(matrix_size[0])*pixel_size[0], line)
           plt.plot(peaks*pixel_size[0], line[peaks], "x")
           plt.xlabel('location [mm]')
           plt.ylabel('intensity')
           plt.show()
    
    if plot:
        plt.figure(figsize=(4,4), dpi=300)
        plt.imshow(np.abs(res), cmap='gray'), plt.axis('off')
        plt.show()
        
        t = ski.filters.threshold_otsu(res)
        plt.figure(figsize=(4,4))
        plt.rc('font', size=12) 
        plt.hist(res.flatten(), bins=100, density=True)
        plt.plot(t,0, 'ro')
        plt.xlabel('intensity')
        plt.ylabel('density')
        plt.show()

    number_of_resolved_line = (np.int8(np.sum(n_res_v)), np.int8(np.sum(n_res_h)))
    
    return pixel_size, number_of_resolved_line

#%% 7. low contrast detectability
def threshold_determination(radius, noise_std, patch_size, fov, matrix_size, n_patch, contrast, plot=False):
    '''
    Determine the threshold for signal present and signal absent patches.

    Parameters
    ----------
    radius : float
        Radius of disk in mm.
    noise_std : float
        standard deviation of noise.
    patch_size : int
        matrix size of the patch.
    fov : tuple
        FOV of the original resolution phantom.
    matrix_size : tuple
        Matrix size of the original resolution phantom.
    n_patch : int
        Number of patches to generate for signal present or signal absent category.
    contrast : float
        Contrast of the resolution phantom.
    plot : boolean, optional
        Plot for debugging. The default is False.

    Returns
    -------
    thre : float
        Threshold for this disk radius, noise, patch size, contrast.
    acc_max : float
        Maximum accuracy corresponding to this threshold.

    '''
    
    # create template
    intensity = 0.5
    _, template = disk_phantom(fov, (radius,radius), matrix_size=matrix_size, intensity=intensity, noise_std=0)
    template = template[matrix_size[0]//2-patch_size:matrix_size[0]//2+patch_size, matrix_size[1]//2-patch_size:matrix_size[1]//2+patch_size]
    template = np.abs(template)
    template = template / np.mean(template) # normalization
    
    # create signal present and signal absent patches
    corr_sp = np.zeros((n_patch,))
    corr_sa = np.zeros((n_patch,))
    
    for n in range(n_patch):
        # create signal present patch
        _, signal_present = disk_phantom(fov, (radius,radius), matrix_size=matrix_size, intensity=contrast, noise_std=noise_std)
        signal_present = signal_present[matrix_size[0]//2-patch_size:matrix_size[0]//2+patch_size, matrix_size[1]//2-patch_size:matrix_size[1]//2+patch_size]
        signal_present = signal_present - np.mean(signal_present) # normalization

        # correlation
        corr_sp[n] = np.amax(sp.signal.correlate2d(signal_present, template, boundary='wrap', mode='same'))
      
    for n in range(n_patch):
        # create signal absent patch with noise
        kspace = np.random.normal(0, noise_std, matrix_size) + 1j * np.random.normal(0, noise_std, matrix_size) 
        # ifft2 compute image
        signal_absent = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace))) * np.sqrt(np.prod(kspace.shape))
        signal_absent = signal_absent[matrix_size[0]//2-patch_size:matrix_size[0]//2+patch_size, matrix_size[1]//2-patch_size:matrix_size[1]//2+patch_size]
        # signal_absent = np.abs(signal_absent)
        signal_absent = (signal_absent - np.mean(signal_absent)) # normalization
        
        # correlation
        corr_sa[n] = np.amax(sp.signal.correlate2d(signal_absent, template, boundary='wrap', mode='same'))
 
    # sweep threshold to find the threshold that yields highest accuracy
    acc_max = 0
    thre_low = 0
    
    for th in np.arange(np.amin(corr_sa), np.amax(corr_sp), 0.01):
        tn_sa = np.sum(corr_sa < th) # true negative
        tp_sp = np.sum(corr_sp > th) # true positive
        acc = (tn_sa + tp_sp) / (len(corr_sa) + len(corr_sp))
        if acc_max < acc:
            acc_max = acc
            thre_low = th
    
    acc_max = 0
    thre_high = 0
    for th in np.arange(np.amax(corr_sp), np.amin(corr_sa), -0.01):
        tn_sa = np.sum(corr_sa < th) # true negative
        tp_sp = np.sum(corr_sp > th) # true positive
        acc = (tn_sa + tp_sp) / (len(corr_sa) + len(corr_sp))
        
        if acc_max < acc:
            acc_max = acc
            thre_high = th
    
    thre = (thre_low + thre_high) / 2
    
    if plot:
        plt.figure()
        plt.subplot(131)
        plt.imshow(template, cmap='gray')
        plt.title('template')
        plt.subplot(132)
        plt.imshow(signal_present, cmap='gray')
        plt.title('signal present')
        plt.subplot(133)
        plt.imshow(signal_absent, cmap='gray')
        plt.title('signal absent')
        
        plt.figure()
        plt.hist(corr_sp, alpha=0.8, bins=50)
        plt.hist(corr_sa, alpha=0.8, bins=50)
        plt.vlines(thre, 0, 20, colors='red', linestyles='dashed')
        plt.legend(['threshold', 'signal present', 'signal absent'])
        plt.title('maximum correlation \n radius={}, noise std={}, contrast={}, accuracy={:.2f}'.format(radius, noise_std, contrast, acc_max))
        plt.show()
    return thre, acc_max

def low_contrast_detectability(lc, fov, radius_range, center, nspokes=10, spoke_dist=20, disk_per_spoke=3, intensity=1, plot=False, contrast=0.3, noise_std=0.0,th=None,acc=None):
    '''
    Low contrast detectability using the low contrast phantom.

    Parameters
    ----------
    lc : ndarray
        Low contrast phantom image.
    fov : tuple
        Field of view in mm.
    radius_range : tuple
        (min_radius, max_radius) of disks.
    center : tuple
        Center of the phantom.
    intensity : float, optional
        Intensity of background. The default is 1.
    nspokes : int, optional
        Number of spokes. The default is 10.
    spoke_dist : float, optional
        Distance between disks within a spoke in mm. The default is 20.
    disk_per_spoke : int, optional
        Number of disks per spoke. The default is 3.
    plot : bool, optional
        Plot the result. The default is True.
    contrast : float, optional
        Contrast of the disks. The default is 0.3.
    noise_std : float, optional
        Standard deviation of the additive Gaussian noise. The default is 0.0.
    thre : float
        Threshold for this disk radius, noise, patch size, contrast.
    acc : float
        Maximum accuracy corresponding to this threshold.

    Returns
    -------
    num_complete_spoke : int
        number of complete spoke, main metric.
    num_corr : ndarray
        number of corralated locations for each disk.

    '''
    
    # matrix_size and pixel size
    matrix_size = np.shape(lc)
    pixel_size = np.array(fov) / np.array(matrix_size)
    num_corr = np.zeros((nspokes, disk_per_spoke))
    
    # compute original disk center in pixel
    theta = 2 * np.pi / nspokes
    center_x = spoke_dist * np.sin(np.arange(nspokes) * theta) / pixel_size[0]
    center_y = spoke_dist * np.cos(np.arange(nspokes) * theta) / pixel_size[1]
    center_x = np.repeat(center_x[:,np.newaxis], disk_per_spoke, axis=1) * np.arange(2,disk_per_spoke+2) + matrix_size[0] / 2
    center_y = np.repeat(center_y[:,np.newaxis], disk_per_spoke, axis=1) * np.arange(2,disk_per_spoke+2) + matrix_size[1] / 2
    center_x = center_x + center[0] / pixel_size[0]
    center_y = center_y + center[1] / pixel_size[1]
    
    # compute the predicted disk size
    disk_radii = np.linspace(radius_range[1], radius_range[0], nspokes)
    disk_radii = np.repeat(disk_radii[:,np.newaxis], disk_per_spoke, axis=1)
    
    # iterate over all original disk center
    corr = np.zeros((nspokes, disk_per_spoke))
    cal_th = 0
    if th is None:
        cal_th = 1
        th = np.zeros((nspokes, disk_per_spoke))
        acc = np.zeros((nspokes, disk_per_spoke))
        
    for m in range(nspokes):
        for n in range(disk_per_spoke):
            # distance to largest correlation location
            cx = np.int16(center_x[m,n])
            cy = np.int16(center_y[m,n])

            # create patch
            patch_size = 5
            patch = lc[cx-patch_size:cx+patch_size, cy-patch_size:cy+patch_size]
            patch = patch - intensity # substract background
            patch = (patch - np.mean(patch))  # normalization
            
            # create disk template
            radius_corr = disk_radii[m,n]
            _, template = disk_phantom(fov, (radius_corr,radius_corr), matrix_size=matrix_size, intensity=0.5)
            template = template[matrix_size[0]//2-patch_size:matrix_size[0]//2+patch_size, matrix_size[1]//2-patch_size:matrix_size[1]//2+patch_size]
            template = np.abs(template)
            template = template / np.mean(template) # normalization
            
            # correlation
            corr_img = sp.signal.correlate2d(patch, template, boundary='wrap', mode='same')
            corr[m, n] = np.amax(corr_img)
            
            if cal_th:
                th[m,n], acc[m,n] = threshold_determination(disk_radii[m,n], noise_std, patch_size, fov, matrix_size, 100, contrast*intensity)
            
            if plot:
                plt.figure(figsize=(12,4), dpi=300)
                plt.subplot(131)
                plt.imshow(template, cmap='gray')
                plt.axis('off')
                plt.title('template')
                
                plt.subplot(132)
                plt.imshow(patch, cmap='gray')
                plt.axis('off')
                plt.title('patch')
                
                plt.subplot(133)
                plt.imshow(corr_img, cmap='gray')
                plt.axis('off')
                plt.title('correlation')
                plt.show()
        
    # if acc < 0.7 disks can not be reliably detected
    signal_present = corr > th
    signal_present[acc < 0.7] = 0
    
    if plot:
        plt.figure(figsize=(4,4), dpi=300)
        plt.imshow(lc, cmap='gray')
        plt.axis('off')
        
        plt.figure(figsize=(4,4), dpi=300)
        plt.imshow(lc, cmap='gray')
        plt.plot(center_y[signal_present], center_x[signal_present], 'ro')
        plt.axis('off')
        plt.show()
            
    # compute the first number of complete spokes
    num_complete_spoke = 0
    for n in range(nspokes):
        if np.sum(signal_present[n] > 0) == disk_per_spoke:
            num_complete_spoke += 1
        else:
            break
        
    return num_complete_spoke, num_corr

