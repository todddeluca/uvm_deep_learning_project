import datetime
import importlib
import keras
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
import SimpleITK # xvertseg MetaImage files
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
import preprocessing
import augmentation



XVERTSEG_MASK_VALS = (200, 210, 220, 230, 240)


def get_mhd_raw_id(d):
    '''
    d: a Path, a directory containing paired MetaImage format files in xVertSeg.
    returns a triple of a list of mhd files, of raw files, and of xvertseg scan ids.
    '''
        
    mhds = [str(p) for p in list(d.glob('*.mhd'))]
    ids = [int(re.search(r'.*?(\d\d\d)\.mhd$', p).group(1)) for p in mhds]
    raws = [d / re.sub(r'\.mhd$', '.raw', p) for p in mhds]
    return mhds, raws, ids


def get_xvertseg_infos(xvertseg_dir):
    '''
    Build a dataframe with columns: id, dataset, image_mhd, image_raw, mask_mhd, mask_raw, and labeled.
    id is the number embedded in the xvertseg filenames.  xvertseg is split into 2 datasets, data1 and data2.
    data1 is labeled, meaning it has segmentation masks.  data2 only has images.
    
    There are 15 labeled images and 10 unlabeled images.
    
    data_dir: the xVertSeg1.v1/Data1 dir, as a Path.
    return: dataframe. 
    '''
    # filename examples
    # image016.mhd
    # image016.raw
    # mask001.mhd
    # mask001.raw
    
    # Data1 has 15 images and masks (labeled data)
    # Data2 has 10 test images with no mask.  Unlabeled data.
    data1_dir = xvertseg_dir / 'Data1'
    idir1 = data1_dir / 'images'
    mdir1 = data1_dir / 'masks'
    data2_dir = xvertseg_dir / 'Data2'
    idir2 = data2_dir / 'images'
    
    img1_mhds, img1_raws, img1_ids = get_mhd_raw_id(idir1)
    img1_df = pd.DataFrame({'id': img1_ids, 'image_mhd': img1_mhds, 'image_raw': img1_raws})
    mask1_mhds, mask1_raws, mask1_ids = get_mhd_raw_id(mdir1)
    mask1_df = pd.DataFrame({'id': mask1_ids, 'mask_mhd': mask1_mhds, 'mask_raw': mask1_raws})
    img2_mhds, img2_raws, img2_ids = get_mhd_raw_id(idir2)
    img2_df = pd.DataFrame({'id': img2_ids, 'image_mhd': img2_mhds, 'image_raw': img2_raws})
    img2_df['dataset'] = ['data2'] * len(img2_df)
    
    infos = img1_df.merge(mask1_df, on='id')
    infos['dataset'] = ['data1'] * len(infos)
    infos = pd.concat([infos, img2_df]).sort_values('id').reset_index(drop=True) # merge data1 and data2.  Neaten id column.
    return infos


def load_xvertseg_img(path):
    '''
    path: an mhd file path.
    returns: a tuple of img nparray (in xyz order of dimensions) and img itk (SimpleITK)
    '''
    # https://github.com/juliandewit/kaggle_ndsb2017/blob/master/step1_preprocess_luna16.py
    itk = SimpleITK.ReadImage(path)
    img_zyx = SimpleITK.GetArrayFromImage(itk)
    img_xyz = np.swapaxes(img_zyx, 0, 2) # swap z and x.

    return img_xyz, itk


def normalize_xvertseg_image(image):
    '''
    img: an xvertseg xyz oriented image.  These images look like they have hounsfield units shifted by +1000 so
    that they will be non-negative numbers.  Anyway...
    
    Normalize each voxel in img by clipping values to lie within 0 to 2000.  
    Scale the numbers to between 0 and 1.
    
    return: normalized image.
    '''
    MIN_BOUND = 0000.0 # Air: -1000, Water: 0 hounsfield units.
    MAX_BOUND = 2000.0 # Bone: 200, 700, 3000.  https://en.wikipedia.org/wiki/Hounsfield_scale
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def plot_image_historgrams():
    '''
    See a histogram of the values in each image in xVertSeg.
    Used to confirm that the values of interest ranged from 0 to 2000, for determining normalization.
    '''
    infos = get_xvertseg_infos(XVERTSEG_DIR)
    for i in range(len(infos)):
        img_zyx, itk = load_xvertseg_img(infos.loc[i, 'image_mhd'])
        img = np.swapaxes(img_zyx, 0, 2) # swap z and x.
        plt.hist(img.ravel(), 256)
        plt.title('image histogram for id ' + str(infos.loc[i, 'id']))
        plt.show()
        

# https://github.com/juliandewit/kaggle_ndsb2017/blob/master/step1_preprocess_luna16.py


def get_preprocessed_xvertseg_image_path(id, preprocessed_dir):
    return str(Path(preprocessed_dir, f'image{id:03}.npy'))


def get_preprocessed_xvertseg_mask_path(id, preprocessed_dir):
    return str(Path(preprocessed_dir, f'mask{id:03}.npy'))


def resample_xvertseg_image(img, spacing, target_spacing, metadata_only=False):

    print('img shape:', img.shape)
    print('img spacing:', spacing)
    print('target spacing:', target_spacing)
    # resample image
    resampled_img, resampled_spacing = preprocessing.resample_image(img, spacing, target_spacing,
                                                                    metadata_only=metadata_only)
    print('resampled image spacing:', resampled_spacing)
    print('resampled image shape:', resampled_img.shape)
    return resampled_img, resampled_spacing


def xvertseg_mask_layers_gen(mask, vals=XVERTSEG_MASK_VALS):
    '''
    Avoid having every mask layer generated at the same time, to save on memory.
    Mask is a categorical mask with categories encoded as numbers specified by vals.
    '''
    for val in vals:
        layer = np.zeros(mask.shape)
        layer[mask == val] = 1.
        yield layer, val
        

def binarize_mask(mask, p=0.5):
    '''
    mask: an array whose value will be thresholded by p.  >p -> 1.  <=p -> 0
    '''
    mask[mask > p] = 1
    mask[mask <= p] = 0
    return mask

    
def resample_xvetseg_mask_layer(layer, spacing, target_spacing, metadata_only=False, p=0.5):
    '''
    p: Binarization threshold.  Everything greater than this theshold is set to 1.  
      Everything less than or equal to p is set to 0.
    '''
    print('resampling...')
    resampled_layer, resampled_spacing = preprocessing.resample_image(
        layer, spacing, target_spacing, metadata_only=metadata_only)
    print('resampled spacing:', resampled_spacing)
    
    resampled_layer = binarize_mask(resampled_layer, p=p)
    return resampled_layer, resampled_spacing


def resample_xvertseg_mask(mask, spacing, target_spacing, metadata_only=False, bin_thresh=0.5):
    '''
    vVertSeg image masks have 6 classes embedded in their values. 
    To be resampled, they need to be split into binary masks, resized, re-binarized, and recombined.
    This is done to minimize the loss of information when resampling blurs voxel values.  There
    is probably a better way.
    
    image: a 3d volume that is an image mask. 
    spacing: the size of a voxel in some units.  E.g. [0.3, 0.3, 0.9]
    target_spacing: the size of a voxel after resampling, in some units.  E.g. [1.0, 1.0, 1.0]
    bin_thresh: binarization threshold.  Used to clean up mask after resampling.
    
    returns: resampled categorical mask with target spacing adjusted because volumes have 
      integer dimensions.
    '''
    resampled_spacing = None
    resampled_mask = None
    # split the image into layers, one layer for each category (except background category).
    for layer, val in xvertseg_mask_layers_gen(mask):
        print('resampling mask layer for val:', val)
        # rlayer is a binary mask, resampled from layer.
        rlayer, rspacing = resample_xvetseg_mask_layer(layer, spacing, target_spacing, p=bin_thresh)
        
        if resampled_spacing is None:
            resampled_spacing = rspacing
            print('resampled_spacing:', resampled_spacing)
            resampled_categorical_mask = np.zeros(rlayer.shape)
            
        if np.any(resampled_spacing != rspacing):
            raise Exception('Resampled spacing did not match previous resampled spacing!', resampled_spacing, rspacing)
            
        # where rlayer and running mask both have data, ignore rlayer data (someone got there first).
        rlayer[(resampled_categorical_mask > 0) & (rlayer > 0)] = 0 
        resampled_categorical_mask = np.add(resampled_categorical_mask, rlayer * val)
            
    return resampled_categorical_mask, resampled_spacing


def _add_triplet_to_infos(triplet, infos, i, col_prefix):
    '''
    For image_dim0, image_dim1, image_dim2 and all the others.
    '''
    for j in range(3):
        infos.loc[i, col_prefix + str(j)] = triplet[j]
        
    return infos
    
    
def preprocess_xvertseg(data_dir, dest_dir, metadata_path, start=0, metadata_only=False,
                        delete_existing=False, bin_thresh=0.5):

    target_spacing = (1.0, 1.0, 1.0)
    preprocessing.delete_and_make_dir(delete_existing, dest_dir)

    # Useful for restarting processing after a failure part-way through.
    if start == 0:
        print('start==0.  getting infos de novo.')
        infos = get_xvertseg_infos(data_dir)
    else:
        print('start > 0. getting infos from metadata cache.')
        infos = read_xvertseg_metadata(metadata_path)
        
    print('info ids')
    print(infos['id'])
    for i in range(start, len(infos)):
        print('info index', i)
        img_path = infos.loc[i, 'image_mhd']
        print('image mhd path:', img_path)
        img, itk = load_xvertseg_img(img_path)
        spacing = np.array(itk.GetSpacing())    # spacing of voxels in world coor. (mm)
        infos = _add_triplet_to_infos(spacing, infos, i, 'image_pixdim')
        infos = _add_triplet_to_infos(img.shape, infos, i, 'image_dim')

        print('img shape:', img.shape)
        print('img spacing:', spacing)

        # resample image
        resampled_img, resampled_spacing = preprocessing.resample_image(
            img, spacing, target_spacing, metadata_only=metadata_only)
        print('resampled image shape:', resampled_img.shape)
        print('resampled image spacing:', resampled_spacing)
        infos = _add_triplet_to_infos(resampled_spacing, infos, i, 'pp_image_pixdim')
        infos = _add_triplet_to_infos(resampled_img.shape, infos, i, 'pp_image_dim')
        
        # Normalize voxel intensities
        if not metadata_only:
            normalized_img = normalize_xvertseg_image(resampled_img)
        
        # save processed image
        path = get_preprocessed_xvertseg_image_path(infos.loc[i, 'id'], dest_dir)
        infos.loc[i, 'pp_image_path'] = str(path)
        print(f'Saving preprocessed image to {path}.')
        if not metadata_only:
            np.save(path, normalized_img)

        # resample mask if it exists
        if pd.notna(infos.loc[i, 'mask_mhd']):
            mask_path = infos.loc[i, 'mask_mhd']
            print('mask mdh path:', mask_path)
            mimg, mitk = load_xvertseg_img(mask_path)
            mask_spacing = np.array(mitk.GetSpacing()) # xyz spacing
            print('mask shape:', mimg.shape)
            print('mask spacing:', mask_spacing)
            infos = _add_triplet_to_infos(mask_spacing, infos, i, 'mask_pixdim')
            infos = _add_triplet_to_infos(mimg.shape, infos, i, 'mask_dim')
            
            if not np.all(mask_spacing == spacing):
                raise Exception('mask_spacing != spacing', mask_spacing, spacing)
                
            rmask, rmspacing = resample_xvertseg_mask(
                mimg, mask_spacing, target_spacing, metadata_only=metadata_only, bin_thresh=bin_thresh)
            print('resampled mask shape:', rmask.shape)
            print('resampled mask spacing:', rmspacing)
            
            if not np.all(rmask.shape == resampled_img.shape):
                raise Exception('resampled mask shape != resampled image shape', rmask.shape, resampled_img.shape)

            infos = _add_triplet_to_infos(rmspacing, infos, i, 'pp_mask_pixdim')
            infos = _add_triplet_to_infos(rmask.shape, infos, i, 'pp_mask_dim')

            path = get_preprocessed_xvertseg_mask_path(infos.loc[i, 'id'], dest_dir)
            infos.loc[i, 'pp_mask_path'] = str(path)
            print(f'Saving preprocessed mask to {path}.')
            if not metadata_only:
                np.save(path, rmask)

        # save metadata every image
        infos = write_xvertseg_metadata(infos, path=metadata_path)
    
    return infos


def write_xvertseg_metadata(infos, path):
    print('saving xvertseg metadata to', path)
    with open(path, 'wb') as fh:
        fh.write(pickle.dumps(infos))
    
    return infos


def read_xvertseg_metadata(path):
    print('reading xvertseg metadata from', path)
    with open(path, 'rb') as fh:
        infos = pickle.loads(fh.read())
    
    return infos



class XvertsegSequence(keras.utils.Sequence):
    '''
    Yields an input image (possibly cropped) and for output one or more of:
    - a binary mask
    - a categorical mask
    - categorical labels (for uncropped images)
    '''

    def __init__(self, infos, batch_size=1, shuffle=True, crop_shape=None, flip=None, 
                 transpose=False, gray_std=None, gray_disco=False):
        '''
        infos: dataframe containing image path and segmentation mask path.
        '''
        self.batch_size = batch_size
        self.shuffle = shuffle

        if shuffle:
            # shuffle x and y with the same shuffled index to preserve pairing of examples and labels.
            shuffled_idx = random.sample(range(len(infos)), k=len(infos))
            self.infos = infos.iloc[shuffled_idx].reset_index(drop=True) # shuffle and zero-index df
        else:
            self.infos = infos.reset_index(drop=True) # zero-index df
        
        # configure outputs and augmentation
        self.config(crop_shape=crop_shape, flip=flip, 
                    transpose=transpose, gray_std=gray_std, gray_disco=gray_disco)
    
    def config(self, batch_size=1, crop_shape=None, flip=None, transpose=False, gray_std=None, gray_disco=False):
        '''
        Configure outputs and augmentation.
        '''
        self.batch_size = batch_size
        self.crop_shape = crop_shape
        self.flip = flip
        self.transpose = transpose
        self.gray_std = gray_std
        self.gray_disco = gray_disco
        return self
        
        
    def __len__(self):
        '''
        Return number of batches, based on batch_size
        '''
        return int(np.ceil(len(self.infos) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        idx: batch index
        '''
#         print(f'Sequence: getting item {idx}')
        
        batch_infos = self.infos.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, :]
#         print('idx', idx, 'batch_size', self.batch_size)
#         print('batch_infos', batch_infos)
        batch_x = []
        batch_y = []
        for i, batch_info in batch_infos.iterrows(): # range(len(batch_infos)):
            
#             img_path = batch_infos.loc[i, 'pp_image_path']
#             mask_path = batch_infos.loc[i, 'pp_mask_path']
            img_path = batch_info['pp_image_path']
            mask_path = batch_info['pp_mask_path']
            img = preprocessing.get_preprocessed_image(img_path)
            mask = binarize_mask(preprocessing.get_preprocessed_image(mask_path)).astype('uint8')
            if not np.all(img.shape == mask.shape):
                raise Exception('image shape != mask shape', img.shape, mask.shape)

            img, mask = augmentation.augment_image_and_mask(
                img, mask, crop_shape=self.crop_shape, gray_std=self.gray_std,
                gray_disco=self.gray_disco, flip=self.flip, transpose=self.transpose)
            
            batch_x.append(np.expand_dims(img, axis=-1))
            batch_y.append(np.expand_dims(mask, axis=-1))
            
        return (np.array(batch_x), np.array(batch_y))
    
    def on_epoch_end(self):
        if self.shuffle:
            shuffled_idx = random.sample(range(len(self.infos)), k=len(self.infos))
            self.infos = self.infos.iloc[shuffled_idx].reset_index(drop=True) # shuffle and zero-index df



def get_xvertseg_datagens(preprocessed_metadata_fun, batch_size=1, seed=None, validation_split=0.25,
                         test_split=None):
    '''
    validation_split: the percentage of the dataset set aside for the validation set.
    test_split: the percentage of the dataset set aside for the test set.  Default: None.
      Specify something other than None to get a (train, val, test) tuple returned.
    
    Each group is split into train and validation (and test) according to validation_split
    and test_split.
    
    Return a tuple of training Sequence and validation Sequence (and test Sequence).
    '''
       
    # gather
    infos = preprocessed_metadata_fun()
    seg_infos = infos[infos['pp_mask_path'].notna()]
    
    # shuffle
    shuffled = seg_infos.sample(frac=1, random_state=seed)
    
    # split
    val_end = int(len(shuffled) * validation_split) # size of val set
    if test_split:
        test_end = int(len(shuffled) * (validation_split + test_split))
    else:
        test_end = val_end
        
    val = shuffled.iloc[:val_end, :].reset_index(drop=True)
    if test_split:
        test = shuffled.iloc[val_end:test_end, :].reset_index(drop=True)
        train = shuffled.iloc[test_end:, :].reset_index(drop=True)
    else:
        train = shuffled.iloc[val_end:, :].reset_index(drop=True)
    print('Train set size:', len(train))
    print('Val set size:', len(val))
    if test_split:
        print('Test set size:', len(test))

    print('Batch size:', batch_size)
    train_gen = XvertsegSequence(train, batch_size=batch_size)
    val_gen = XvertsegSequence(val, batch_size=batch_size)
    if test_split:
        test_gen = XvertsegSequence(test, batch_size=batch_size)
    print('Num train batches:', len(train_gen))
    print('Num val batches:', len(val_gen))
    
    if test_split:
        return train_gen, val_gen, test_gen
    else:
        return train_gen, val_gen



def main():
    '''
    Example of how to preprocess the dataset.
    '''
    # preprocess images and masks
    infos = preprocess_xvertseg()
    
    
if __name__ == '__main__':
    main()
