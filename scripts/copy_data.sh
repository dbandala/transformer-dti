#!/bin/bash
# -*- coding: utf-8 -*-
# Author: Daniel Bandala @ nov 2022
# copy_data.sh HCP 69

# Input variables
main_root=$1
# validate input variables
if [ -z "$main_root" ]; then
    echo "Missing arguments (Eg. copy_data.sh HCP)" && exit -1
fi

# Check for each restaurant
cd $main_root || exit -1

# create folder if it does not exist
if [ -d ~/Documents/dti_data ]; then
    rm -rf ~/Documents/dti_data
fi
mkdir ~/Documents/dti_data
if [ ! -d ~/Documents/dti_data/$main_root ]; then
    mkdir ~/Documents/dti_data/$main_root
    mkdir ~/Documents/dti_data/$main_root/case_{01..35}
fi

# copy each case data
for case_folder in $(ls -d case_*); do
    # copy input folder   
    cp -rf $case_folder/input ~/Documents/dti_data/$main_root/$case_folder/input
    # copy output data
    cp -rf $case_folder/dti ~/Documents/dti_data/$main_root/$case_folder/dti
done
