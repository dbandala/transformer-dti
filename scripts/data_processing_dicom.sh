#!/bin/bash
# -*- coding: utf-8 -*-
# Author: Daniel Bandala @ oct 2022
# data_processing.sh HCP 1

# Input variables
main_root=$1
extract_shells=$2
# Script variables
dwi_folder=dwi
output_folder=dti
middleproc=middleproc
mask_threeshold=0.2
phase_direction=AP
# Check for each restaurant
cd $main_root || exit -1
for case_folder in $(ls -d *); do
    # change working directory to data folder
    cd $case_folder || exit -1
    # enter second folder
    for data_folder in $(ls -d *); do
        # change directory
        cd $data_folder || exit -1
        # delete previous processing folder
        # if [ -d $dwi_folder ]; then
        #     rm -rf $dwi_folder
        # fi
        if [ -d $middleproc ]; then
            rm -rf $middleproc
        fi
        # create output folders
        if [ ! -d $dwi_folder ]; then
            mkdir $dwi_folder
        fi
        if [ ! -d $output_folder ]; then
            mkdir $output_folder
        fi
        mkdir $middleproc
        # find folder
        for dti_folder in $(ls -d *DTI*); do
            # subfolders variables
            out_folder=$output_folder/$dti_folder
            dwi_f=$dwi_folder/$dti_folder
            # check if dwi input has been generated
            if [ ! -d "$dwi_f" ]; then
                mkdir $dwi_f
                # convert DICOM to NIFTI images
                if [ -f "comments.txt" ]; then
                    mrconvert $dti_folder/ $dwi_f/DWI.nii.gz -export_grad_mrtrix $dwi_f/grad_table.txt -export_grad_fsl $dwi_f/bvecs.txt $dwi_f/bvals.txt -force -set_property comments "$(cat comments.txt)"
                else
                    mrconvert $dti_folder/ $dwi_f/DWI.nii.gz -export_grad_mrtrix $dwi_f/grad_table.txt -export_grad_fsl $dwi_f/bvecs.txt $dwi_f/bvals.txt -force
                fi
            fi
            # check if data already processed
            if [ -d "$out_folder" ]; then
                echo "Data already processed"
                continue
            fi
            # output log
            echo
            echo "Starting to process $case_folder/$data_folder/$dti_folder ..."
            echo "Current directory: $PWD"
            # create processing folder
            mkdir $out_folder
            
            # display mrinfo
            mrinfo $dti_folder/
            # start DWI-DTI processing
            ini_file=$middleproc/DWI.mif
            # convert DICOM images to MIF format
            if [ -f "comments.txt" ]; then
                mrconvert $dti_folder/ -set_property comments "$(cat comments.txt)" $ini_file -force
            else
                mrconvert $dti_folder/ $ini_file -force
            fi

            # extract shells if needed
            if [ ! -z "$extract_shells" ] && [ $extract_shells -eq 1 ]; then
                mrconvert $ini_file $middleproc/DWI_single.mif -coord 3 0:1:68 -force
                #dwiextract -shells 0,1000 $ini_file $pf/DWI_shells.mif -force
                #dwiextract -singleshell -bzero $pf/DWI_shells.mif $pf/DWI_single.mif -force
                ini_file=$middleproc/DWI_single.mif
            fi

            # extract noise from DWI images
            dwidenoise $ini_file $middleproc/DWI_denoised.mif -noise $middleproc/DWI_noise.mif -force -estimator Exp2

            # calculate residual noise
            mrcalc $ini_file $middleproc/DWI_denoised.mif -subtract $middleproc/residual.mif -force

            # remove gibbs rings
            mrdegibbs $middleproc/DWI_denoised.mif $middleproc/DWI_no_rgibbs.mif

            # eddy current and motion correction
            dwifslpreproc $middleproc/DWI_no_rgibbs.mif $middleproc/DWI_preproc.mif -rpe_none -pe_dir $phase_direction -force

            # convert to NII files and export gradient table and corrected bvals and bvecs
            mrconvert $middleproc/DWI_preproc.mif $middleproc/DATA.nii.gz -export_grad_mrtrix $middleproc/grad_table.txt -export_grad_fsl $middleproc/bvecs $middleproc/bvals -force

            # extract brain mask and binary mask
            bet2 $middleproc/DATA.nii.gz $middleproc/BET.nii.gz -f $mask_threeshold -m

            # fit diffusion tensor
            dtifit --data=$middleproc/DATA.nii.gz --out=$out_folder/DTI --mask=$middleproc/BET.nii.gz_mask.nii.gz --bvecs=$middleproc/bvecs --bvals=$middleproc/bvals --save_tensor

            # print line breaks
            printf "\n\n"
        done
        # delete preprocessing folder
        rm -rf $middleproc
        # return to database folder
        cd ..
    done
    cd ..
done
