#!/usr/bin/env bash -l


#
# AI tools for perinatal brain MRI analysis
#
# Copyright 2026 - King's College London
#
# The auto SVRTK code and all scripts are distributed under the terms of the
# [GNU General Public License v3.0: 
# https://www.gnu.org/licenses/gpl-3.0.en.html. 
# 
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation version 3 of the License. 
# 
# This software is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the GNU General Public License for more details.
#


src=/home/neonatal-low-field-mri-analysis
mirtk=/bin/MIRTK/build/lib/tools


org_t1=$1
proc=$2
out_lab=$3


if [[ $# -ne 3 ]] ; then

    echo
    echo "------------------------------------------------------------"
    echo
    echo "Usage: please use the following format ..."
    echo "bash /home/neonatal-low-field-mri-analysis/scripts/run-3t-t1w-neo-brain-bet-extraction-2026.sh  [full_path_to_input_t1w_image.nii.gz] [full_path_to_folder_for_tmp_processing] [full_path_to_output_label.nii.gz]"
    echo
    echo "------------------------------------------------------------"
    echo
    exit

fi 



echo 
echo "------------------------------------------------------------"
echo
echo " - SCRIPT FOR NEONATAL BRAIN EXTRACTION PROCESSING ... "
echo
echo "------------------------------------------------------------"
echo 
  


echo
echo "------------------------------------------------------------"
echo
echo " - input t2 : " ${org_t1}
echo " - processing folder : " ${proc}
echo
echo "------------------------------------------------------------"
echo
echo " - RUNNING PREPROCESSING ... "
echo
echo "------------------------------------------------------------"
echo

if [[ ! -f ${org_t1} ]];then
    echo
    echo "------------------------------------------------------------"
    echo
    echo "ERROR: NO INPUT FILE ..."
    echo
    echo "------------------------------------------------------------"
    echo
    exit
fi

if [[ ! -d ${proc} ]];then
    mkdir ${proc}
fi

if [[ ! -d ${proc} ]];then
    echo
    echo "------------------------------------------------------------"
    echo
    echo "ERROR: CANNOT CREATE PROCESSING FOLDER ..."
    echo
    echo "------------------------------------------------------------"
    echo
    exit
fi

 
 
${mirtk}/convert-image ${org_t1} ${proc}/org-t1.nii.gz

${mirtk}/extract-image-region ${proc}/org-t1.nii.gz ${proc}/org-t1.nii.gz -Rt1 0 -Rt2 0

${mirtk}/nan ${proc}/org-t1.nii.gz 100000

${mirtk}/threshold-image ${proc}/org-t1.nii.gz ${proc}/m-t1.nii.gz 0.5 > ${proc}/t.txt

${mirtk}/extract-connected-components ${proc}/m-t1.nii.gz ${proc}/m-t1.nii.gz

${mirtk}/crop-image ${proc}/org-t1.nii.gz ${proc}/m-t1.nii.gz ${proc}/crop-tr-t1.nii.gz

${mirtk}/pad-3d ${proc}/crop-tr-t1.nii.gz ${proc}/pad-crop-tr-t1-128.nii.gz 128 1

echo
echo " - brain extraction ..."
echo


w_bet=${src}/models/at_unet_bet_brain_neo_3t_t1_2026_128_best_metric_model.pth

#unset PYTHONPATH ;
python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-cpu.py 128 1 ${w_bet} ${proc}/pad-crop-tr-t1-128.nii.gz ${proc}/bet-lab-pad-crop-t1-128.nii.gz

${mirtk}/extract-connected-components ${proc}/bet-lab-pad-crop-t1-128.nii.gz ${proc}/bet-lab-pad-crop-t1-128.nii.gz

${mirtk}/dilate-image ${proc}/bet-lab-pad-crop-t1-128.nii.gz ${proc}/dl-bet-lab-pad-crop-t1-128.nii.gz -iterations 1

${mirtk}/erode-image ${proc}/bet-lab-pad-crop-t1-128.nii.gz ${proc}/dl-bet-lab-pad-crop-t1-128.nii.gz -iterations 1

${mirtk}/transform-image ${proc}/bet-lab-pad-crop-t1-128.nii.gz ${out_lab} -labels -target ${proc}/org-t1.nii.gz


if [[ ! -f ${out_lab} ]];then
    echo 
    echo "------------------------------------------------------------"
    echo
    echo "ERROR - LABEL FILE WAS NOT GENERATED ..."
    echo 
    echo "------------------------------------------------------------"
    echo
    exit
    
else

    # rm -r 

    echo
    echo "------------------------------------------------------------"
    echo
    echo " - output label : " ${out_lab}
    echo
    echo "------------------------------------------------------------"


fi


chmod 777 -R ${proc} ${out_lab}

 
