#!/bin/bash
# -*- coding: utf-8 -*-
# Author: Daniel Bandala @ oct 2022
# input_generation.sh HCP

# Input variables
main_root=$1
#total_volumes=$2 [ -z "$total_volumes" ]
# validate input variables
if [ -z "$main_root" ]; then
    echo "Missing arguments (Eg. input_generation.sh HCP)" && exit -1
fi
# Initial variables
df=input
pf=preproc
#step=$((total_volumes/6))
start=0
end=35 #68 #$((step*6-1))
# Check for each restaurant
cd $main_root || exit -1
for case_folder in $(ls -d case_*); do
    # check if data already processed
    if [ -d $case_folder/$df ]; then
        rm -rf $case_folder/$df
        #continue
    fi
    # change working directory to case folder
    cd $case_folder
    # output log
    echo "Starting to generate inputs for $case_folder ..."
    # delete previous processing folder
    if [ -d $pf ]; then
        rm -rf $pf
    fi
    # create processing folders
    mkdir $df
    mkdir $pf
    
    # start processing
    ini_file=$pf/DWI.mif
    # convert DWI images to MIF format
    mrconvert -fslgrad bvecs.txt bvals.txt DWI.nii.gz $ini_file -force

    # extract signals
    mrconvert $ini_file $pf/DWI_signals.mif -coord 3 $start:$end -force

    # extract noise from DWI images
    #dwidenoise $pf/DWI_signals.mif $pf/DWI_denoised.mif -noise $pf/DWI_noise.mif -force -estimator Exp2

    # convert to NII files and export gradient table and extracted bvals and bvecs
    mrconvert $pf/DWI_signals.mif $df/DWI.nii.gz -export_grad_mrtrix $df/grad_table.txt -export_grad_fsl $df/bvecs.txt $df/bvals.txt -force

    # delete preprocessing folder
    rm -rf $pf
    # print line breaks
    printf "\n\n"
    # return to database folder
    cd ..
done
