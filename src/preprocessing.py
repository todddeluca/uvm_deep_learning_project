import datetime
import importlib
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import projd
import random
import re
import scipy
import shutil
import sys
from sklearn.model_selection import train_test_split
import uuid

import matplotlib.pyplot as plt # data viz
import seaborn as sns # data viz

import imageio # display animated volumes
from IPython.display import Image # display animated volumes

from IPython.display import SVG # visualize model
from keras.utils.vis_utils import model_to_dot # visualize model

import util


def get_image(path):
    # read the image from the filesystem
    img = nib.load(path).get_data()
    return img
   
    
def get_preprocessed_image(path):
    return np.load(path)
    

def resample_image(image, spacing, new_spacing, metadata_only=False):
    '''
    image: a 3d volume
    spacing: the size of a voxel in some units.  E.g. [0.3, 0.3, 0.9]
    new_spacing: the size of a voxel after resampling, in some units.  E.g. [1.0, 1.0, 1.0]
    
    returns: resampled image and new spacing adjusted because images have integer dimensions.
    '''
    # calculate resize factor required to change image to new shape
    spacing = np.array(spacing)
    new_spacing = np.array(new_spacing)
    spacing_resize_factor = spacing / new_spacing
    new_real_shape = image.shape * spacing_resize_factor
    new_shape = np.round(new_real_shape).astype(int)
    real_resize_factor = new_shape / image.shape
    
    # adjusted spacing to account for integer dimensions of resized image.
    new_spacing = spacing / real_resize_factor
    
    if metadata_only:
        new_image = np.zeros(new_shape)
    else:
        new_image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        
    return new_image, new_spacing


def normalize_nifti_image(image):
    '''
    Normalize voxel units by clipping them to lie between -1000 and 1000 hounsfield units 
    and then scale number to between 0 and 1.
    '''
    MIN_BOUND = -1000.0 # Air: -1000, Water: 0 hounsfield units.
    MAX_BOUND = 1000.0 # Bone: 200, 700, 3000.  https://en.wikipedia.org/wiki/Hounsfield_scale
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def get_preprocessed_image_path(scan_id, preprocessed_dir):
    return str(Path(preprocessed_dir, f'{scan_id}.npy'))


def delete_and_make_dir(delete_existing, directory):
    if delete_existing and directory.is_dir():
        print('Removing existing dir:', directory)
        shutil.rmtree(directory)
    if not directory.exists():
        print('Making directory:', directory)
        directory.mkdir(parents=True)
    
    
def preprocess_nifti_scans(normals_dir, fractures_dir, dest_dir, metadata_path, delete_existing=False, metadata_only=False):
    '''
    metadata_only: if True, scan images are not preprocessed and saved to disk, but metadata is generated as if it had been.
      This is useful for quickly regenerating metadata that does not require reprocessing all files, 
      like when adding `pp_pixdim0`, etc. to metadata.
    '''
    
    delete_and_make_dir(delete_existing, dest_dir)
    
    # get all scan infos
    normal_infos = util.get_nifti_infos(util.get_nifti_files(normals_dir))
    fracture_infos = util.get_nifti_infos(util.get_nifti_files(fractures_dir))

    # add class label
    normal_infos['class'] = 'normal'
    fracture_infos['class'] = 'fracture'
    
    # process and save each image.
    infos = pd.concat([normal_infos, fracture_infos]).reset_index(drop=True) # index from 0:len(infos)

    for i in range(len(infos)):
        print('image index:', i)
        info = infos.loc[i, :]
        img_path = str(info['path'])
        print('image path:', img_path)
        scan_id = info['id']
        print('image id:', scan_id)
        if metadata_only:
            # dummy image of the right shape, to speed processing of metadata
            img = np.zeros((info['dim0'], info['dim1'], info['dim2']))
        else:
            img = get_image(img_path)
        print('image shape:', img.shape)
        
        # Standardize voxel size to 1mm^3 to reduce image size.
        spacing = (info['pixdim0'], info['pixdim1'], info['pixdim2'])
        target_spacing = (1.0, 1.0, 1.0)
        print('image spacing:', spacing)
        print('new spacing:', target_spacing)
        resampled_img, resampled_spacing = resample_image(img, spacing, target_spacing, metadata_only=metadata_only)
        print('resampled image spacing:', resampled_spacing)
        print('resampled image shape:', resampled_img.shape)
        
        # Normalize voxel intensities
        if not metadata_only:
            normalized_img = normalize_image(resampled_img)
            print('Normalized image shape:', normalized_img.shape)
        
        # save processed image
        path = get_preprocessed_image_path(scan_id, dest_dir)
        print(f'Saving preprocessed image to {path}.')
        if not metadata_only:
            np.save(path, normalized_img)
        
        # track image metadata
        infos.loc[i, 'pp_path'] = str(path)
        # voxel dimensions
        infos.loc[i, 'pp_pixdim0'] = resampled_spacing[0] 
        infos.loc[i, 'pp_pixdim1'] = resampled_spacing[1] 
        infos.loc[i, 'pp_pixdim2'] = resampled_spacing[2] 
        # image dimensions
        infos.loc[i, 'pp_dim0'] = resampled_img.shape[0]
        infos.loc[i, 'pp_dim1'] = resampled_img.shape[1]
        infos.loc[i, 'pp_dim2'] = resampled_img.shape[2]
        

    # save metadata
    write_preprocessed_metadata(infos, path=metadata_path)
    return infos
    
        
def write_preprocessed_metadata(infos, path):
    print('saving preproccessed metadata to', path)
    with open(path, 'wb') as fh:
        fh.write(pickle.dumps(infos))
    
    return infos


def read_preprocessed_metadata(path, filter='uvmmc'):
    print('reading preproccessed metadata from', path)
    with open(path, 'rb') as fh:
        infos = pickle.loads(fh.read())
    
    if filter == 'uvmmc':
        # drop bad data.  
        # 082222_190 input spacing and shape are [0.9765625, 0.9765625, 0.625] and [512, 307, 1]
        infos = infos[infos['id'] != '082222_190']
        
    return infos


def main():
    '''
    Example of how to preprocess the dataset.
    '''
    SEED = 0
    EPOCHS = 10
    BATCH_SIZE = 1
    PATCH_SHAPE = (32, 32, 32)

    MODEL_NAME = 'model_01'

    DATA_DIR = Path('/data2').expanduser()
    NORMAL_SCANS_DIR = DATA_DIR / 'uvmmc/nifti_normals'
    FRACTURE_SCANS_DIR = DATA_DIR / 'uvmmc/nifti_fractures'
    PROJECT_DATA_DIR = DATA_DIR / 'uvm_deep_learning_project'
    PP_IMG_DIR = PROJECT_DATA_DIR / 'uvmmc' / 'preprocessed' # preprocessed scans dir
    PP_MD_PATH = PROJECT_DATA_DIR / 'uvmmc' / 'preprocessed_metadata.pkl'
    TMP_DIR = DATA_DIR / 'tmp'

    for d in [DATA_DIR, NORMAL_SCANS_DIR, PROJECT_DATA_DIR, PP_IMG_DIR, TMP_DIR, PP_MD_PATH.parent]:
        if not d.exists():
            d.mkdir(parents=True)

    # Uncomment to preprocess images
    infos = preprocess_nifti_scans(NORMAL_SCANS_DIR, FRACTURE_SCANS_DIR, dest_dir=PP_IMG_DIR, metadata_path=PP_MD_PATH, metadata_only=True)
    
    
if __name__ == '__main__':
    main()

