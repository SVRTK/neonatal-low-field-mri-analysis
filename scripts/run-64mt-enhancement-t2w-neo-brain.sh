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


org_t2=$1
out_folder=$2
out_img=$3
out_org=$4
out_lab=$5



src=/home/neonatal-low-field-mri-analysis
proc=/home/tmp_proc



if [[ $# -ne 5 ]] ; then

    echo
    echo "------------------------------------------------------------"
    echo
    echo "Usage: please use the following format ..."
    echo "bash /home/neonatal-low-field-mri-analysis/run-64mt-enhancement-t2w-neo-brain.sh [full_path_to_input_64mt_.nii.gz] [full_path_to_folder_for_out_qc] [full_path_to_reoriented_output_enhanced_image.nii.gz] [full_path_to_reoriented_original_image.nii.gz] [full_path_to_output_label.nii.gz]"
    echo
    echo "------------------------------------------------------------"
    echo
    exit

fi 



echo 
echo "------------------------------------------------------------"
echo
echo " - SCRIPT FOR ANATOMY-AWARE 64mT T2W IMAGE ENHANCEMENT ... "
echo
echo "------------------------------------------------------------"
echo 
  


echo
echo "------------------------------------------------------------"
echo
echo " - input t2 : " ${org_t2}
#echo " - processing folder : " ${proc}
echo
echo "------------------------------------------------------------"
echo
echo " - RUNNING PREPROCESSING ... "
echo
echo "------------------------------------------------------------"
echo

if [[ ! -f ${org_t2} ]];then
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
else
    rm -r ${proc}/*
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


if [[ ! -d ${out_folder} ]];then
    mkdir ${out_folder}
fi

if [[ ! -d ${out_folder} ]];then
    echo
    echo "------------------------------------------------------------"
    echo
    echo "ERROR: CANNOT CREATE OUTPUT FOLDER ..."
    echo
    echo "------------------------------------------------------------"
    echo
    exit
fi



${mirtk}/convert-image ${org_t2} ${proc}/org-t2.nii.gz

${mirtk}/nan ${proc}/org-t2.nii.gz 900000

${mirtk}/pad-3d ${proc}/org-t2.nii.gz ${proc}/pad-t2-128.nii.gz 128 1

echo
echo " - brain extraction ..."
echo

w_bet=${src}/models/at_unet_bet_neo_brain_64mt_t2_hilo_best_metric_model.pth

#unset PYTHONPATH ;
python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-cpu.py 128 1 ${w_bet} ${proc}/pad-t2-128.nii.gz ${proc}/bet-lab-pad-t2-128.nii.gz

${mirtk}/extract-connected-components ${proc}/bet-lab-pad-t2-128.nii.gz ${proc}/bet-lab-pad-t2-128.nii.gz

${mirtk}/dilate-image ${proc}/bet-lab-pad-t2-128.nii.gz ${proc}/dl-bet-lab-pad-t2-128.nii.gz -iterations 1

${mirtk}/transform-image ${proc}/bet-lab-pad-t2-128.nii.gz ${proc}/bet-lab-org-t2.nii.gz -labels -target ${proc}/org-t2.nii.gz

${mirtk}/crop-image ${proc}/org-t2.nii.gz ${proc}/dl-bet-lab-pad-t2-128.nii.gz ${proc}/masked-crop-tr-t2.nii.gz

${mirtk}/edit-image ${src}/templates/ref-64mt-neo/dl-ref-t2-64mt.nii.gz ${proc}/tmp-dl-ref.nii.gz -copy-origin ${proc}/masked-crop-tr-t2.nii.gz

${mirtk}/transform-image ${proc}/masked-crop-tr-t2.nii.gz ${proc}/masked-crop-tr-t2.nii.gz -target ${proc}/tmp-dl-ref.nii.gz

${mirtk}/crop-image ${proc}/masked-crop-tr-t2.nii.gz ${proc}/dl-bet-lab-pad-t2-128.nii.gz ${proc}/masked-crop-tr-t2.nii.gz

${mirtk}/mask-image ${proc}/masked-crop-tr-t2.nii.gz  ${proc}/dl-bet-lab-pad-t2-128.nii.gz ${proc}/masked-crop-tr-t2.nii.gz

${mirtk}/nan ${proc}/masked-crop-tr-t2.nii.gz 900000

${mirtk}/pad-3d ${proc}/masked-crop-tr-t2.nii.gz ${proc}/pad-masked-crop-tr-t2-128.nii.gz 128 1




echo
echo " - l/r extraction & reorientation ..."
echo

w_lr=${src}/models/at_unet_lr_brain_64mt_t2_hilo_2lab_best_metric_model.pth

#unset PYTHONPATH ;
python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-cpu.py  128 2 ${w_lr} ${proc}/pad-masked-crop-tr-t2-128.nii.gz ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz


${mirtk}/edit-image ${src}/templates/ref-64mt-neo/dl-ref-t2-64mt.nii.gz ${proc}/tmp-dl-ref.nii.gz -copy-origin ${proc}/masked-crop-tr-t2.nii.gz

${mirtk}/edit-image ${src}/templates/ref-64mt-neo/lr-lab-64mt.nii.gz ${proc}/tmp-lr-ref.nii.gz -copy-origin ${proc}/masked-crop-tr-t2.nii.gz

${mirtk}/mask-image ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz ${proc}/bet-lab-org-t2.nii.gz  ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz


${mirtk}/register ${proc}/tmp-lr-ref.nii.gz ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz -model Affine -dofin ${src}/templates/ref-64mt-neo/i.dof -dofout ${proc}/aff-d.dof -v 0

${mirtk}/init-dof ${proc}/i-reset.dof -dofin ${proc}/aff-d.dof -sx 100 -sy 100 -sz 100  -sxy 0 -syz 0 -sxz 0

${mirtk}/register ${proc}/tmp-lr-ref.nii.gz ${proc}/lr-lab-pad-masked-crop-t2-128.nii.gz -model Rigid -dofin ${proc}/i-reset.dof -dofout ${proc}/r-d.dof -v 0

${mirtk}/resample-image ${proc}/bet-lab-org-t2.nii.gz ${proc}/dl-tmp.nii.gz -size 1 1 1 -interp NN

${mirtk}/dilate-image ${proc}/dl-tmp.nii.gz ${proc}/dl-tmp.nii.gz -iterations 2

${mirtk}/mask-image ${proc}/org-t2.nii.gz ${proc}/dl-tmp.nii.gz ${proc}/masked-crop-org-t2.nii.gz

${mirtk}/crop-image ${proc}/masked-crop-org-t2.nii.gz ${proc}/dl-tmp.nii.gz ${proc}/masked-crop-org-t2.nii.gz
 
${mirtk}/transform-image ${proc}/masked-crop-org-t2.nii.gz ${proc}/reo-masked-t2.nii.gz  -target ${proc}/tmp-dl-ref.nii.gz -dofin ${proc}/aff-d.dof

${mirtk}/threshold-image ${proc}/reo-masked-t2.nii.gz ${proc}/m.nii.gz 0.5 > ${proc}/t.txt

${mirtk}/crop-image ${proc}/reo-masked-t2.nii.gz ${proc}/m.nii.gz ${proc}/reo-masked-t2.nii.gz

${mirtk}/pad-3d ${proc}/reo-masked-t2.nii.gz ${proc}/pad-reo-masked-t2-128.nii.gz 128 1

${mirtk}/transform-image ${proc}/org-t2.nii.gz ${proc}/pad-reo-masked-t2-128.nii.gz -target ${proc}/pad-reo-masked-t2-128.nii.gz -dofin ${proc}/aff-d.dof

${mirtk}/transform-image ${proc}/bet-lab-org-t2.nii.gz ${proc}/pad-reo-bet-128.nii.gz -target ${proc}/pad-reo-masked-t2-128.nii.gz -labels -dofin ${proc}/aff-d.dof

${mirtk}/dilate-image ${proc}/pad-reo-bet-128.nii.gz ${proc}/pad-reo-bet-128.nii.gz

${mirtk}/mask-image ${proc}/pad-reo-masked-t2-128.nii.gz ${proc}/pad-reo-bet-128.nii.gz ${proc}/pad-reo-masked-t2-128.nii.gz

${mirtk}/nan ${proc}/pad-reo-masked-t2-128.nii.gz 900000



echo
echo "------------------------------------------------------------"
echo
echo " - RUNNING ENHANCEMENT ... "
echo
echo "------------------------------------------------------------"
echo


w_enhancement=${src}/models/basic_unet_hilo_gan_localhist_t2_052026_ssim_l1_edge_bounti_contrpres64mt_15lab_best_metric_model.pth

python3 ${src}/src/run_monai_basiunet_enhancement_1case-2026-cpu.py ${w_enhancement} ${proc}/pad-reo-masked-t2-128.nii.gz ${proc}/pad-reo-bet-128.nii.gz ${proc}/enhanced-reo-t2.nii.gz ${proc}/multi-lab-reo-t2.nii.gz ${proc}/qc-segmentation.nii.gz ${proc}/qc-prediction.nii.gz ${proc}/qc-contrast.nii.gz ${proc}/qc-summary.csv

${mirtk}/match-histogram ${proc}/pad-reo-masked-t2-128.nii.gz ${proc}/enhanced-reo-t2.nii.gz ${proc}/enhanced-reo-t2.nii.gz -Tp 0 -Sp 0  > ${proc}/tmp.txt

${mirtk}/invert-dof ${proc}/aff-d.dof ${proc}/inv-aff-d.dof

${mirtk}/compose-dofs ${proc}/r-d.dof ${proc}/inv-aff-d.dof ${proc}/from-aff-to-rigid-reo.dof

${mirtk}/edit-image ${src}/templates/ref-64mt-neo/ref-082025-64mt-1mm.nii.gz ${proc}/tmp-final-1mm-ref.nii.gz -copy-origin ${proc}/masked-crop-org-t2.nii.gz


${mirtk}/transform-image ${proc}/multi-lab-reo-t2.nii.gz ${proc}/tmp-final-ref-masked.nii.gz -target  ${proc}/tmp-final-1mm-ref.nii.gz  -dofin ${proc}/from-aff-to-rigid-reo.dof -labels

${mirtk}/dilate-image ${proc}/tmp-final-ref-masked.nii.gz ${proc}/tmp-final-ref-masked.nii.gz -iterations 2

${mirtk}/crop-image ${proc}/tmp-final-ref-masked.nii.gz ${proc}/tmp-final-ref-masked.nii.gz ${proc}/tmp-final-ref-masked.nii.gz

${mirtk}/transform-image ${proc}/enhanced-reo-t2.nii.gz ${out_img} -target ${proc}/tmp-final-ref-masked.nii.gz  -dofin ${proc}/from-aff-to-rigid-reo.dof -interp BSpline

${mirtk}/transform-image ${proc}/multi-lab-reo-t2.nii.gz ${out_lab} -target ${proc}/tmp-final-ref-masked.nii.gz  -dofin ${proc}/from-aff-to-rigid-reo.dof -labels

${mirtk}/nan ${out_img} 1000000



cp ${proc}/r-d.dof ${out_folder}/from-org-to-rigid-reo.dof

${mirtk}/transform-image  ${proc}/qc-segmentation.nii.gz ${out_folder}/qc-segmentation.nii.gz -target  ${proc}/tmp-final-ref-masked.nii.gz  -dofin ${proc}/from-aff-to-rigid-reo.dof

${mirtk}/transform-image  ${proc}/qc-prediction.nii.gz ${out_folder}/qc-prediction.nii.gz -target  ${proc}/tmp-final-ref-masked.nii.gz  -dofin ${proc}/from-aff-to-rigid-reo.dof

${mirtk}/transform-image  ${proc}/qc-contrast.nii.gz ${out_folder}/qc-contrast.nii.gz -target  ${proc}/tmp-final-ref-masked.nii.gz  -dofin ${proc}/from-aff-to-rigid-reo.dof

${mirtk}/transform-image  ${proc}/org-t2.nii.gz ${out_org} -target  ${proc}/tmp-final-ref-masked.nii.gz  -dofin ${proc}/r-d.dof

cp ${proc}/qc-summary.csv ${out_folder}/

cp ${proc}/bet-lab-org-t2.nii.gz ${out_folder}/



if [[ ! -f ${out_img} ]];then
    echo
    echo "------------------------------------------------------------"
    echo
    echo "ERROR - FILES WERE NOT GENERATED ..."
    echo
    echo "------------------------------------------------------------"
    echo
    exit
    
else

    echo
    echo "------------------------------------------------------------"
    echo
    echo " - output enhanced reoriented image : " ${out_img}
    echo " - output original reoriented image : " ${out_org}
    echo " - output processing folder with QC : " ${out_folder}
    echo " - output brain tissue parcellation : " ${out_lab}
    echo
    echo "------------------------------------------------------------"


fi


chmod 777 -R ${out_img} ${out_lab} ${out_folder}

