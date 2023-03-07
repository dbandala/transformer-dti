#!/usr/bin/env python
# coding: utf-8
# Daniel Bandala @ feb-2023
import sys,os,glob
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from scipy import spatial

# optimum directions
GRAD_DIR = [
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [-1,0,0],
    [0,-1,0],
    [0,0,-1],
    [0.57,0.57,0.57],
    [0.57,-0.57,0.57],
    [-0.57,-0.57,0.57],
    [-0.57,0.57,0.57],
    [0.57,0.57,-0.57],
    [0.57,-0.57,-0.57],
    [-0.57,-0.57,-0.57],
    [-0.57,0.57,-0.57]
]

def extract_volumes(main_dir):
    # get cases list
    path_list = glob.glob(os.path.join(main_dir,'case_*'))
    # for each case extract optimum directions
    for path in path_list:
        # print case
        print("Processing ", os.path.basename(path))
        #input path
        input = os.path.join(path,'input')
        # read bvecs data
        bvecs_file = os.path.join(input,'bvecs.txt')
        with open(bvecs_file) as file:
            bvecs = [line.split() for line in file]
        # rearrange vectors
        bvecs = [list(i) for i in zip(*bvecs)]
        # fit tree to vectors
        tree = spatial.KDTree(bvecs)
        # read difussion data
        data, affine = load_nifti(os.path.join(input,'DWI.nii.gz'))
        #data = np.random.rand(16,16,10,69)
        # get closest directions
        best_dir = []
        for grad in GRAD_DIR:
            best_dir = best_dir+[tree.query(grad)[1]]
        # extract only the optimum grad directions
        data = data[:,:,:,best_dir]
        # save new file
        save_nifti(os.path.join(input,'DWI_opt.nii.gz'), data, affine)
        # complete print
        print("Complete")

if __name__ == "__main__":
    args = sys.argv
    if len(args)<2:
        print("Arguments error. Script call: python extract_grad_directions.py path/to/data")
        sys.exit()
    # create main directory path
    current_directory = os.getcwd()
    main_dir = os.path.join(current_directory,args[1])
    # call extract function
    extract_volumes(main_dir)
