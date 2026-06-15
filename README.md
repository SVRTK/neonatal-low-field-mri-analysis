Automated analysis tools for neonatal MRI at 64mT
====================

This repository contains DL pipelines for [MONAI](https://github.com/Project-MONAI/MONAI)-based automated analysis neonatal brain MRI.


- The repository, scripts and models were designed and created at the Department of Early Life Imaging, King's College London.

  
- Please email alena.uus (at) kcl.ac.uk if in case of any questions.


**- IMPORTANT NOTES:**

**- this is a new methods and we would be very grateful for your feedback so that it can be improved! Please email us.**

**- the current version of the pipeline was trained on specific Hyperfine 64mT acquisition protocol (Cawley, 2023) only - it might not work on other acquisitions.**



Development of these processing and analysis tools was supported by projects led by Prof Mary Rutherford, Prof Tomoki Arichi, Prof 
Jonathan O'Muircheartaigh, Prof David Edwards and Prof Jo Hajnal.



<img src="info/multi-bounti-3t-full.jpg" alt="AUTOSVRTKEXAMPLE" height="400" align ="center" />




Auto processing scripts 
------------------------


**The automated docker tags are _fetalsvrtk/svrtk:neonatal_low_field_mri_analysis_amd_ (AMD system) _fetalsvrtk/svrtk:neonatal_low_field_mri_analysis_amd_ (ARM system)**


**ANATOMY-AWARE T2w CONTRAST ENHANCEMENT FOR 64mT NEONATAL BRAIN MRI:**

*Input data requirements:*
- sufficient image quality, no extreme artifacts
- good quality 3D SVR
- fetal TE=250ms - dHCP protocol 
- full ROI coverage
- standard radiological space
- 25-50 weeks scan PMA
- no extreme structural anomalies (the network was not trained on too many abnormal cases)

*Outputs:*

- **!!! the output resolution will be 1.0mm**
  
Note: you will need >16GB GPU for -gpu option


**PLEASE RUN IT DIRECTLY VIA OUR DOCKER:**

_Note: for MAC - please use docker pull fetalsvrtk/svrtk:perinatal_brain_mri_analysis_arm and CPU version _


```bash

docker pull fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd

# contrast enhancement + segmentation / surface extraction: CPU version 
docker run --rm --gpus all --mount type=bind,source=LOCATION_ON_YOUR_MACHINE,target=/home/data  fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd sh -c ' bash /home/neonatal-low-field-mri-analysis/scripts/run-64mt-enhancement-t2w-neo-brain-cpu.sh [/home/data/path_to_input_t2w_64mt_image.nii.gz] [/home/data/path_to_output_processing_folder] [/home/data/path_to_output_reo_enhanced_image.nii.gz] [/home/data/path_to_output_reo_original_image.nii.gz]  [/home/data/path_to_output_reo_tissue_label.nii.gz]  ; '

# contrast enhancement + segmentation / surface extraction: GPU version 
docker run --rm --gpus all --mount type=bind,source=LOCATION_ON_YOUR_MACHINE,target=/home/data  fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd sh -c ' bash /home/neonatal-low-field-mri-analysis/scripts/run-64mt-enhancement-t2w-neo-brain-gpu.sh [/home/data/path_to_input_t2w_64mt_image.nii.gz] [/home/data/path_to_output_processing_folder] [/home/data/path_to_output_reo_enhanced_image.nii.gz] [/home/data/path_to_output_reo_original_image.nii.gz]  [/home/data/path_to_output_reo_tissue_label.nii.gz]  ; '


```


**PROCESSING EXAMPLE:**

```bash

docker run --rm --gpus all --mount type=bind,source=/home/au18/folder_with_datasets,target=/home/data  fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd sh -c ' bash /home/neonatal-low-field-mri-analysis/scripts/run-64mt-enhancement-t2w-neo-brain-cpu.sh /home/data/input-t2w-64mt.nii.gz  /home/data/proc-outputs /home/data/output-reo-enhanced.nii.gz  /home/data/output-reo-orignal.nii.gz  /home/data/output-reo-lab.nii.gz   ; '

```




License
-------

The code and model weights are distributed under the terms of the
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation version 3 of the License. 

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.


Citation and acknowledgements
-----------------------------

In case you found this repository useful please give appropriate credit to the software.


**64mT contrast enhancement:**
> Uus, A., Fukami-Gartner, A., Kyriakopoulou, V., Cromb, D., Morgan, T., Arulkumaran, S., Egloff Collado, A., Luis, A., Bos, R., Makropoulos, A., Schuh, A., Robinson, E., Sousa, H., Deprez, M., Cordero-Grande, L., Bradshaw, C., Colford, K., Hutter, J., Price, A., O’Muircheartaigh, J., Hammers, A., Rueckert, D., Counsell, S., McAlonan, G., Arichi, T., Edwards, A. D., Hajnal, J. V., Rutherford, M. A., Story, L. (2026). Multi-BOUNTI: Multi-lobe Brain vOlUmetry and segmeNtation for feTal and neonatal MRI. medRxiv, 2026.04.21.26351376. https://doi.org/10.64898/2026.04.21.26351376






Disclaimer
-------

This software has been developed for research purposes only, and hence should not be used as a diagnostic tool. In no event shall the authors or distributors be liable to any direct, indirect, special, incidental, or consequential damages arising of the use of this software, its documentation, or any derivatives thereof, even if the authors have been advised of the possibility of such damage.

