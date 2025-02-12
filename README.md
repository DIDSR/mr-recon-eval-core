# MR Recon Eval: Digital Image Quality Phantoms and Automated Evaluation Metrics for Assessing Machine Learning-Based MRI Reconstruction


<img src="https://github.com/DIDSR/mr-recon-eval-core/assets/37195678/e8eedb5d-75b4-4b70-991c-c71e2af018e1" width="800">

## Summary
This repository contains the open-source Python code for the paper titled "Evaluating Machine Learning-Based MRI Reconstruction Using Digital Image Quality Phantoms". It consists of:

1. Digital phantom creation (digital_phantom.py): creating 3 types of phantoms in k-space: disk, resolution, low-contrast phantom
2. Metrics evaluation (evaluation_metrics.py): geometric accuracy, intensity uniformity, percentage ghosting, sharpness, SNR, high contrast resolution, and low contrast detectability.



Disclaimer
----------

This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.

## Start Here

1. Install python 3.11.3 or any version greater (Note: the code was tested on python 3.11.3 and 3.11.5 by the authors but maybe compatible with a lower version of python.)

2. Clone this repository and navigate to its root directory

3. Install the required dependencies in a python enviroment 

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
Run demo.py in a python enviroment.

**Expected Output**

<img src="https://github.com/DIDSR/mr-recon-eval-core/assets/162378584/385842cf-2eca-46ea-ab8e-a1b7eebf3bbc" width="200">
<img src="https://github.com/DIDSR/mr-recon-eval-core/assets/162378584/82f16b3e-b37e-45fa-abd6-c01b010c04a4" width="200">
<img src="https://github.com/DIDSR/mr-recon-eval-core/assets/162378584/58300a3b-b9f5-4450-9fc6-e9755edd3490" width="200">

```
Geometric accuracy (maximum percentage error): 0.0003
Intensity uniformity: 94.2322
Percentage ghosting: 0.0009
Sharpness fwhm: 1.5268 Sharpness slope: 0.3233
SNR dual: 11.3576
High contrast resolution:
         number of resolved vertical lines in the upper resolution block:  4
         number of resolved horizontal lines in the lower resolution block:  4
Calculating low contrast detectability measurement (may take a while) ...
Low contrast detectability (number of complete spokes):  9
```

**The exact values will be different in each realization due to random noise in the generated phantom images. The generated phantom images will be saved to an ouptut folder.

## Reference

- **How to cite** 
   Tan F, Delfino JG, Zeng R. Evaluating Machine Learning-Based MRI Reconstruction Using Digital Image Quality Phantoms. Bioengineering. 2024; 11(6):614. https://doi.org/10.3390/bioengineering11060614

## Contacts

Rongping Zeng, rongping.zeng@fda.hhs.gov
Fei Tan, fei.tan@fda.hhs.gov  
