# MR Recon Eval: Digital Image Quality Phantoms and Automated Evaluation Metrics for Assessing Machine Learning-Based MRI Reconstruction

placeholder for link to paper

<img src="https://github.com/fei-tan-fda/mr-recon-eval/assets/162378584/04af8d97-4be4-419b-9c00-c7845bd5bead" width="500">

## Summary
This repository contains the open-source Python code for the paper titled "Evaluating Machine Learning-Based MRI Reconstruction Using Digital Image Quality Phantoms". It consists of:

1. Digital phantom creation (digital_phantom.py): creating 3 types of phantoms in k-space: disk, resolution, low-contrast phantom
2. Metrics evaluation (evaluation_metrics.py): geometric accuracy, intensity uniformity, percentage ghosting, sharpness, SNR, high contrast resolution, and low contrast detectability.

**Contacts:**

Rongping Zeng, rongping.zeng@fda.hhs.gov
Fei Tan, fei.tan@fda.hhs.gov  

Disclaimer
----------

This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.

## Start Here

1. Install python 3.11.3 or any version greater (Note: the code was tested on python 3.11.3 and 3.11.5 by the authors but maybe compatible with a lower version ofpython.)

2. Clone this repository and navigate to its root directory

3. Install the required dependencies 

   ```pip install numpy matplotlib scikit-image scipy spyder```


<!-- 
**if using virtual enviroment**
Create a virtual environtment named eval (or any name of your choosing) 

```python -m venv <chosen_env_name>```

Activate the environment (Ensure you replace <chosen_env_name> with your chosen venv name)

Windows: ```<chosen_env_name>\Scripts\activate```
Unix\Linux: ```source <chosen_env_name>/bin/activate```

Install the required dependencies 

```pip install numpy matplotlib scikit-image scipy spyder```

To deactivate

```deactivate```

**If using Anaconda**

Create conda environment

```conda create -n eval python=3.11.5```

Activate conda environment

```conda activate eval```

Install dependencies

```conda install numpy matplotlib scikit-image scipy spyder``` -->


## Demo
Run Demo in Spyder

Run ```spyder``` and then open the code "demo.py" to run the demo code.

**Expected Output**

<img src="https://github.com/DIDSR/mr-recon-eval-core/assets/162378584/385842cf-2eca-46ea-ab8e-a1b7eebf3bbc" width="200">
<img src="https://github.com/DIDSR/mr-recon-eval-core/assets/162378584/82f16b3e-b37e-45fa-abd6-c01b010c04a4" width="200">
<img src="https://github.com/DIDSR/mr-recon-eval-core/assets/162378584/58300a3b-b9f5-4450-9fc6-e9755edd3490" width="200">

```
geometric accuracy, maximum percentage error:  0.0004787829883421062
intensity uniformity:  94.42621651886563
percentage ghosting:  0.0014833648906810717
sharpness fwhm:  1.7792694914919003 sharpness slope:  0.2886013210608282
SNR dual:  9.686461713187088
high contrast resolution, number of resolved line:  (4, 4)
Calculating low contrast detectability measurement ...
low contrast detectability number of complete spokes:  9
```

**The exact numbers will be different due to random noise in each realization.

## Contribute

- **How to cite** 
   Fei Tan, Jana Delfino, Rongping Zeng, "Evaluating Machine Learning-Based MRI Reconstruction Using Digital Image Quality Phantoms". BioEngineering, Volxx, Pgxx, 2024 (Under review)

## Key Functions: Create Digital Phantom (digital_phantom.py)
Creates disk, resolution, low-contrast phantom.


### disk_phantom
**Description:** Creates a disk phantom.

**Parameters:**
- `fov` (tuple): Field of View (FOV) in millimeters (mm).
- `radius` (tuple): Circle radius in millimeters (mm).
- `center` (tuple, optional): Center coordinates of the disk in millimeters (mm). Default is (0,0).
- `theta` (float, optional): Rotation angle in degrees. Default is 0.
- `matrix_size` (tuple, optional): Matrix size in number of pixels. Default is (64,64).
- `intensity` (float, optional): Intensity of the disk. Default is 1.
- `noise_std` (float, optional): Standard deviation of complex Gaussian noise. Default is 0.

**Returns:**
- `kspace` (ndarray): Complex k-space representation of the disk.
- `disk` (ndarray): Image of the disk.

**Reference:** [Digital_Reference_Objects GitHub repository](https://github.com/philips-labs/Digital_Reference_Objects/blob/main/DRO.py)

### resolution_phantom
**Description:** Creates a resolution phantom.

**Parameters:**
- `fov` (tuple): Field of view in millimeters (mm).
- `radius` (tuple): Radii of individual holes in millimeters (mm).
- `center` (tuple, optional): Center of the phantom in millimeters (mm). Default is (0,0).
- `array` (tuple, optional): Number of holes in x,y dimension. Default is (4,4).
- `matrix_size` (tuple, optional): Matrix size of k-space & image. Default is (64,64).
- `intensity` (float, optional): Intensity of the image. Default is 1.
- `noise_std` (float, optional): Standard deviation of additive complex Gaussian noise. Default is 0.1.

**Returns:**
- `kspace_res` (ndarray): Complex k-space.
- `res` (ndarray): Image of resolution phantom.

### low_contrast_phantom
**Description:** Creates a low contrast phantom.

**Parameters:**
- `fov` (tuple): Field of view in millimeters (mm).
- `radius_range` (tuple): Smallest and largest radii in millimeters (mm).
- `center` (tuple, optional): Center of the phantom in millimeters (mm). Default is (0,0).
- `nspokes` (int, optional): Number of spokes. Default is 10.
- `spoke_dist` (float, optional): Distance between phantoms within one spoke. Default is 20.
- `disk_per_spoke` (int, optional): Number of disks per spoke. Default is 3.
- `matrix_size` (tuple, optional): Matrix size. Default is (64,64).
- `intensity` (float, optional): Intensity of background phantom. Default is 1.
- `contrast` (float, optional): Additive intensity of the foreground phantoms. Default is 0.1.
- `noise_std` (float, optional): Noise standard deviation. Default is 0.1.

**Returns:**
- `kspace_lc` (ndarray): K-space of low contrast phantom.
- `lc` (ndarray): Image of low contrast phantom.

## Key Functions: Evaluation Metrics (`evaluation_metrics.py`)

### geometric_accuracy
**Description:** Measure the geometric accuracy of a disk phantom.

**Parameters:**
- `disk` (ndarray): (M, N) Image to measure.
- `fov` (tuple): Field of view in millimeters (mm) (fov_x, fov_y).
- `radius` (tuple): Ground truth radius in millimeters (mm).
- `center` (tuple): Ground truth center in millimeters (mm).
- `plot` (bool, optional): Plot predicted axes, center, and bounding box. Default is False.

**Returns:**
- `max_percentage_error` (float): Maximum percentage error of major axis and minor axis compared with ground truth.
- `eccentricity` (float): Equals to 0 when input image is circle, range [0, 1).

### intensity_uniformity
**Description:** Measure the intensity uniformity using a disk phantom.

**Parameters:**
- `disk` (ndarray): (M, N) Image to measure.
- `fov` (tuple): Field of view in millimeters (mm) (fov_x, fov_y).
- `radius` (tuple): Ground truth radius in millimeters (mm).
- `center` (tuple): Ground truth center in millimeters (mm).
- `intensity` (float): Ground truth intensity.
- `plot` (bool, optional): Plot predicted axes, center, and bounding box. Default is False.
- `location_known` (bool, optional): Consider the center of the disk known or unknown. Default is True. If known, use the known center to define ROI. If unknown, use thresholding to detect ROI.

**Returns:**
- `intensity_bias` (float): Intensity bias, percentage error of mean intensity compared with ground truth.
- `intensity_uniform` (float): Intensity uniformity.


### percentage_ghosting
**Description:** Measure the percentage ghosting of a disk phantom.

**Parameters:**
- `disk` (ndarray): (M, N) Image to measure.
- `fov` (tuple): Field of view in millimeters (mm) (fov_x, fov_y).
- `center` (tuple): Ground truth center in millimeters (mm).
- `intensity` (float): Ground truth intensity.
- `plot` (bool, optional): Plot predicted axes, center, and bounding box. Default is False.

**Returns:**
- `ghosting_ratio` (float): Ghosting ratio, percentage image ghosting.

### sharpness
**Description:** Sharpness using the edge spread function of a disk phantom.

**Parameters:**
- `disk` (ndarray): Disk image.
- `fov` (tuple): FOV of the disk.
- `radius` (tuple): Radius of the disk.
- `center` (tuple): Center of the disk.
- `plot` (bool, optional): Plot figures for debugging. Default is False.
- `fit` (bool, optional): Whether to fit the edge spread function to sigmoid. Default is False.

**Returns:**
- `fwhm` (float): Full-width-half-maximum of the fitted Lorentzian function.
- `mal_val` (float): Maximum value of the fitted Lorentzian function.



### snr_dual_image
**Description:** SNR measurement using two disks.

**Parameters:**
- `disk1` (ndarray): 1st disk phantom.
- `disk2` (ndarray): 2nd disk phantom.
- `fov` (tuple): FOV of the disk phantoms in mm.
- `radius` (tuple): Radius of the disk phantoms in mm.
- `center` (tuple): Center of the disk phantoms in mm.
- `plot` (bool, optional): Plot for debugging. Default is False.
- `signal_roi_width` (int, optional): Signal ROI width. Default is 7.
- `noise_roi_width` (int, optional): Noise ROI width. Default is 13.

**Returns:**
- `snr` (float): SNR.



### high_contrast_resolution
**Description:** High contrast resolution measurement using resolution phantom.

**Parameters:**
- `res` (ndarray): Image of resolution phantom.
- `fov` (tuple): Field of view in mm.
- `radius` (tuple): Radii of individual holes in mm.
- `center` (tuple, optional): Center of the phantom in mm. Default is (0,0).
- `array` (tuple, optional): Number of holes in x,y dimension. Default is (4,4).
- `plot` (bool, optional): Plot for debugging. Default is True.
  
**Returns:**
- `pixel_size` (tuple): Pixel size in both directions.
- `number_of_resolved_line` (tuple): Number of resolved lines in both directions.

### threshold_determination
**Description:** Determine the threshold for signal present and signal absent patches.
**Parameters:**
- `radius` (float): Radius of disk in mm.
- `noise_std` (float): Standard deviation of noise.
- `patch_size` (int): Matrix size of the patch.
- `fov` (tuple): FOV of the original resolution phantom.
- `matrix_size` (tuple): Matrix size of the original resolution phantom.
- `n_patch` (int): Number of patches to generate for signal present or signal absent category.
- `contrast` (float): Contrast of the resolution phantom.
- `plot` (bool, optional): Plot for debugging. Default is False.
  
**Returns:**
- `thre` (float): Threshold for this disk radius, noise, patch size, contrast.
- `acc_max` (float): Maximum accuracy corresponding to this threshold.

### low_contrast_detectability
**Description:** Low contrast detectability using the low contrast phantom.

**Parameters:**
- `lc` (ndarray): Low contrast phantom image.
- `fov` (tuple): Field of view in millimeters (mm).
- `radius_range` (tuple): (min_radius, max_radius) of disks.
- `center` (tuple): Center of the phantom.
- `intensity` (float, optional): Intensity of background. Default is 1.
- `nspokes` (int, optional): Number of spokes. Default is 10.
- `spoke_dist` (float, optional): Distance between disks within a spoke in millimeters (mm). Default is 20.
- `disk_per_spoke` (int, optional): Number of disks per spoke. Default is 3.
- `plot` (bool, optional): Plot the result. Default is True.
- `contrast` (float, optional): Contrast of the disks. Default is 0.3.
- `noise_std` (float, optional): Standard deviation of the additive Gaussian noise. Default is 0.0.
  
**Returns:**
- `num_complete_spoke` (int): Number of complete spokes, main metric.
- `num_corr` (ndarray): Number of correlated locations for each disk.
