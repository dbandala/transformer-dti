#!/bin/bash
# -*- coding: utf-8 -*-
# Author: Daniel Bandala @ oct 2022
# data_processing.sh HCP 1

# Input variables
main_root=$1
extract_shells=$2
# Initial variables
df=dti
pf=preproc
mask_threeshold=0.2
phase_direction=AP
# Check for each restaurant
cd $main_root || exit -1
for case_folder in $(ls -d case_*); do
    # check if data already processed
    if [ -d "$case_folder/dti" ]; then
        continue
    fi
    # change working directory to case folder
    cd $case_folder
    # output log
    echo "Starting to process $case_folder ..."
    # delete previous processing folder
    if [ -d $pf ]; then
        rm -rf $pf
    fi
    # create processing folders
    mkdir $df
    mkdir $pf
    
    # start DWI-DTI processing
    ini_file=$pf/DWI.mif
    # convert DWI images to MIF format
    mrconvert -fslgrad bvecs.txt bvals.txt DWI.nii.gz $ini_file -force

    # extract shells if needed
    if [ ! -z "$extract_shells" ] && [ $extract_shells -eq 1 ]; then
    	mrconvert $ini_file $pf/DWI_single.mif -coord 3 0:1:68 -force
        #dwiextract -shells 0,1000 $ini_file $pf/DWI_shells.mif -force
        #dwiextract -singleshell -bzero $pf/DWI_shells.mif $pf/DWI_single.mif -force
        ini_file=$pf/DWI_single.mif
    fi

    # extract noise from DWI images
    dwidenoise $ini_file $pf/DWI_denoised.mif -noise $pf/DWI_noise.mif -force -estimator Exp2

    # calculate residual noise
    mrcalc $ini_file $pf/DWI_denoised.mif -subtract $pf/residual.mif -force

    # remove gibbs rings
    mrdegibbs $pf/DWI_denoised.mif $pf/DWI_no_rgibbs.mif

    # eddy current and motion correction
    dwifslpreproc $pf/DWI_no_rgibbs.mif $pf/DWI_preproc.mif -rpe_none -pe_dir $phase_direction -force

    # convert to NII files and export gradient table and corrected bvals and bvecs
    mrconvert $pf/DWI_preproc.mif $pf/DATA.nii.gz -export_grad_mrtrix $pf/grad_table.txt -export_grad_fsl $pf/bvecs $pf/bvals -force

    # extract brain mask and binary mask
    bet2 $pf/DATA.nii.gz $pf/BET.nii.gz -f $mask_threeshold -m

    # fit diffusion tensor
    dtifit --data=$pf/DATA.nii.gz --out=dti/DTI --mask=$pf/BET.nii.gz_mask.nii.gz --bvecs=$pf/bvecs --bvals=$pf/bvals --save_tensor

    # delete preprocessing folder
    rm -rf $pf
    # print line breaks
    printf "\n\n"
    # return to database folder
    cd ..
done
