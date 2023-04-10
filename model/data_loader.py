#!/usr/bin/env python
# coding: utf-8
# Daniel Bandala @ nov-2022
import random,os,glob,torch
import numpy as np
from dipy.io.image import load_nifti
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
# training tools
from sklearn.model_selection import train_test_split 

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def freeze_network(model, layer_name):
    for name, p in model.named_parameters():
        if layer_name in name:
            p.requires_grad = False

def data_preprocessing(img_path, maps=["FA"], signals=12, dti_folder='output'):
    # read difussion data
    data,_ = load_nifti(os.path.join(img_path,'input','DWI_opt.nii.gz'))
    # reshape vectors
    h, w, s, c = data.shape
    data =  np.array(data.transpose(2,3,0,1),dtype=np.float32)
    # select only the first 'signals' channels
    data = data[:,range(signals),:,:]
    # read output tensor data
    if len(maps)==0 or len(maps)>1:
        output = np.zeros((s,len(maps),h,w),dtype=np.float32) if not len(maps)==0 else np.zeros((len(maps)+6,s,h,w),dtype=np.float32)
        for i in range(len(maps)):
            output_aux,_ = load_nifti(os.path.join(img_path,dti_folder,'DTI_'+maps[i]+'.nii.gz'))
            output[:,i,:,:] = np.array(output_aux.transpose(2,0,1),dtype=np.float32)
    else:
        output,_ = load_nifti(os.path.join(img_path,dti_folder,'DTI_'+maps[0]+'.nii.gz'))
        output = np.array(output.transpose(2,0,1),dtype=np.float32)
    if len(maps)==0:
        output_aux,_ = load_nifti(os.path.join(img_path,dti_folder,'DTI_tensor.nii.gz'))
        output[i:] = np.array(output_aux.transpose(2,3,0,1),dtype=np.float32)
    # numpay array to tensor
    data = torch.tensor(data)
    output = torch.tensor(output)
    # normilize data
    data = normalize_data(data)
    output = normalize_data(output)
    # conditioning data
    data, output = data_conditioning(data, output)
    # return tensors
    return data, output

def data_preprocessing_slice(img_path, slice_idx, maps=["FA"], signals=12, dti_folder='output'):
    # read difussion data
    data,_ = load_nifti(os.path.join(img_path,'input','DWI_opt.nii.gz'))
    # read output tensor data
    output,_ = load_nifti(os.path.join(img_path,dti_folder,'DTI_FA.nii.gz'))
    # reshape vectors
    h, w, s, c = data.shape
    data =  np.array(data.transpose(2,3,0,1)[slice_idx],dtype=np.float32)
    output = np.array(output.transpose(2,0,1)[slice_idx],dtype=np.float32)
    # select only the first 'signals' channels
    data = data[:,range(signals),:,:]
    # numpay array to tensor
    data = torch.tensor(data)
    output = torch.tensor(output)
    # normilize data
    data = normalize_data(data)
    output = normalize_data(output)
    # conditioning data
    data, output = data_conditioning(data, output)
    # return data tensors
    return data, output

def input_preprocessing(img_path, signals=12):
    # read difussion data
    data,_ = load_nifti(os.path.join(img_path,'input','DWI_opt.nii.gz'))
    # reshape vectors
    data = np.array(data.transpose(2,3,0,1),dtype=np.float32)
    # select only the first 'signals' channels
    data = data[:,range(signals),:,:]
    # numpay array to tensor
    data = torch.tensor(data)
    return data

# Define class to manage diffussion dataset 
class DWIDataset(Dataset):
    def __init__(self, file_list, maps=["FA"], signals=12, patch_stride:int=20, data_augmentation=False, dti_folder='output'):
        self.file_list = file_list
        self.maps = maps
        self.signals = signals
        self.patch_stride = patch_stride
        self.data_augmentation = data_augmentation
        self.dti_folder = dti_folder
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    def __getitem__(self, idx):
        # preprocessing function
        data, output = data_preprocessing(self.file_list[idx], maps=self.maps, signals=self.signals, dti_folder=self.dti_folder)
        # data augmentation
        if self.data_augmentation:
            data, output = data_augmentation(data, output)
        # patches generation
        #data, output = patches(data, output, stride=self.patch_stride)
        # return data
        return data, output

class DWIDatasetSlice(Dataset):
    def __init__(self, file_list, maps=["FA"], signals=12, patch_stride:int=20,  data_augmentation=False, dti_folder='output'):
        self.file_list = file_list
        self.maps = maps
        self.signals = signals
        self.patch_stride = patch_stride
        self.data_augmentation = data_augmentation
        self.dti_folder = dti_folder
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    def __getitem__(self, idx):
        # retrieve data information
        img_path, slice_idx = self.file_list[idx].split('|')
        slice_idx = int(slice_idx)
        # preprocessing function
        data, output = data_preprocessing_slice(img_path=img_path, slice_idx=slice_idx, maps=self.maps, signals=self.signals, dti_folder=self.dti_folder)
        # data augmentation
        if self.data_augmentation:
            data, output = data_augmentation(data, output)
        # patches generation
        # data, output = patches(data, output, stride=self.patch_stride)
        # return data
        return data, output

def get_data(main_dir, test_size=0.1):
    main_dir = [main_dir] if isinstance(main_dir, str) else main_dir
    # get all dataset paths
    data_list = []
    for dir in main_dir:
        data_list = data_list + glob.glob(os.path.join(dir,'case_*'))
    # printing length of the dataset
    print(f"Dataset size: {len(data_list)}")
    # split train and validation subsets
    train_list, valid_list = train_test_split(data_list, 
                                            test_size=test_size,
                                            random_state=16)
    print(f"Train Data: {len(train_list)}")
    print(f"Validation Data: {len(valid_list)}")
    # return trainning and validation sets
    return train_list, valid_list

def get_data_slice(main_dir, test_size=0.1):
    main_dir = [main_dir] if isinstance(main_dir, str) else main_dir
    # get all dataset paths
    data_list = []
    for dir in main_dir:
        data_list = data_list + glob.glob(os.path.join(dir,'case_*'))
    # printing length of the dataset
    print(f"Dataset size: {len(data_list)}")
    # split train and validation subsets
    train_list_ini, valid_list = train_test_split(data_list, 
                                            test_size=test_size,
                                            random_state=16)
    print(f"Train Data: {len(train_list_ini)}")
    print(f"Validation Data: {len(valid_list)}")
    # generate slide info for each trainning case
    train_list = []
    for te in train_list_ini:
        data,_ = load_nifti(os.path.join(te,'input','DWI_opt.nii.gz'))
        _,_,slides,_ = data.shape
        for j in range(slides):
            train_list.append(te+"|"+str(j))
    # shuffle again trainning list
    random.shuffle(train_list)
    # return trainning and validation sets
    return train_list, valid_list

def get_dataset(main_dir, test_size=0.1, data_steps=4, slice_mode=False, maps=["FA"], signals=12, dti_folder='output'):
    train_list, valid_list = get_data_slice(main_dir, test_size) if slice_mode else get_data(main_dir, test_size)
    #loading dataloader
    train_loader = DataLoader(dataset=DWIDataset(train_list, maps, signals, data_augmentation=True, dti_folder=dti_folder), batch_size=len(train_list)//data_steps, shuffle=True)
    valid_loader = DataLoader(dataset=DWIDataset(valid_list, maps, signals, dti_folder=dti_folder), batch_size=1, shuffle=True)
    return train_list, valid_list, train_loader, valid_loader

# Data conditioning
def data_conditioning(img, label, def_size=140, def_slices=96):
    # data dimensions
    size = img.size()
    spat_dim = size[-1]
    slices = size[0]
    signals = size[1]
    # resize image
    if spat_dim!=def_size:        
        img = TF.resize(img, (def_size, def_size))
        label = TF.resize(label, (def_size, def_size))
    # append extra slices
    if slices<def_slices:
        slc_add = (def_slices-slices)
        slc = torch.zeros((slc_add,signals,def_size,def_size))
        img = torch.cat((img, slc), 0)
        slc = torch.zeros((slc_add,def_size,def_size))
        label = torch.cat((label, slc), 0)
    # return data tensors
    return img, label

# Data augmentation
def data_augmentation(img, label):
    # Random horizontal flipping
    if random.random() > 0.5:
        img = TF.hflip(img)
        label = TF.hflip(label)
    # Random vertical flipping
    if random.random() > 0.5:
        img = TF.vflip(img)
        label = TF.vflip(label)
    # Random rotation
    if random.random() > 0.5:
        angle = random.choice([90,180,270])
        img = TF.rotate(img, angle)
        label = TF.rotate(label, angle)
    # return data tensors
    return img, label

# data normalization
def normalize_data(img):
    max_value = torch.max(img)
    # return max normalization
    return img/max_value

# image patches generation
def patches(img, out, size=35, stride=35):
    # patch input images
    img_shape = img.shape
    img = img.permute(1,0,2,3)
    img_patches = img.unfold(2, size, stride).unfold(3, size, stride)
    img_patches = img_patches.reshape(img_shape[1], -1, size, size)
    img_patches = img_patches.permute(1,0,2,3)
    # patch output images
    out_patches = out.unfold(1, size, stride).unfold(2, size, stride)
    out_patches = out_patches.reshape(-1, size, size)
    return img_patches, out_patches

# reconstruction of data from patches
def data_reconstruction(patches, slices=96, size=140, patch_size=35):
    img = patches.reshape(slices, -1, patch_size, patch_size)
    pr =  size//patch_size
    img = img.reshape(slices, pr, pr, patch_size, patch_size)
    img = img.permute(0, 1, 3, 2, 4).contiguous().view(slices, size, size)
    return img

# get trainable parameters    
def get_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
