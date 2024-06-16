#!/bin/bash
# -*- coding: utf-8 -*-
# Author: Daniel Bandala @ may 2024
# data_processing.sh HCP 1

# Check if the user provided the folder path
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

# Check if the folder exists
if [ ! -d "$1" ]; then
    echo "Error: Folder $1 not found."
    exit 1
fi

# Change directory to the specified folder
cd "$1" || exit

# Loop through all .zip files in the folder
for file in *.zip; do
    echo "Extracting $file ..."
    unzip "$file"
done

echo "Extraction complete."