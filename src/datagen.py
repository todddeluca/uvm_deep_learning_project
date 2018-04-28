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


def random_crop(img, shape):
    '''
    Randomly crop an image to a shape.  Location is chosen at random from
    all possible crops of the given shape.
    
    img: a volume to crop
    shape: size of cropped volume.  e.g. (32, 32, 32)
    '''
    assert all(img.shape[i] >= shape[i] for i in range(len(shape)))
    
    # if img.shape[i] == 32 and shape[i] == 32, i_max == 0.
    maxes = [img.shape[i] - shape[i] for i in range(len(shape))]
    # the starting corner of the crop
    starts = [random.randint(0, m) for m in maxes]
    # Will this indexing work?
    cropped_img = img[[slice(starts[i], starts[i] + shape[i]) for i in range(len(shape))]]
    return cropped_img
        

def augment_image(img, crop_shape=None):
    if crop_shape:
        return random_crop(img, crop_shape)
    else:
        return img


class NiftiSequence(keras.utils.Sequence):

    def __init__(self, x_infos, batch_size, crop_shape, shuffle=True):
        '''
        x_paths: list of paths to preprocessed images
        '''
        if shuffle:
            self.x = x_infos.sample(frac=1).reset_index(drop=True) # shuffle and zero-index df
        else:
            self.x = x_infos.reset_index(drop=True) # zero-index df
            
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_shape = crop_shape
        # assert len(self.x) == len(self.y)

    def __len__(self):
        '''
        Return number of batches, based on batch_size
        '''
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        idx: batch index
        '''
        # print(f'NiftiSequence: getting item {idx}')
        
        # loc indexing uses inclusive name-based indexing, I know I know don't ask, hence the -1.
        batch_x_paths = list(self.x.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'pp_path'])
        # add channel dimension to each augmented (randomly cropped) image.
        batch_x = [np.expand_dims(augment_image(preprocessing.get_preprocessed_image(path), 
                                                crop_shape=self.crop_shape), axis=-1)
                   for path in batch_x_paths]

        # return x and y batches
        return (np.array(batch_x), np.array(batch_x))
    
    def on_epoch_end(self):
        if self.shuffle:
            # print('on_epoch_end: Shuffling sequence')
            self.x = self.x.sample(frac=1).reset_index(drop=True)

            
class FractureSequence(keras.utils.Sequence):

    def __init__(self, x_infos, y_infos, batch_size=1, shuffle=True):
        '''
        x_paths: list of paths to preprocessed images
        '''
        assert len(x_infos) == len(y_infos)

        self.batch_size = batch_size
        self.shuffle = shuffle

        if shuffle:
            # shuffle x and y with the same shuffled index to preserve pairing of examples and labels.
            shuffled_idx = random.sample(range(len(x_infos)), k=len(x_infos))
            self.x = x_infos.iloc[shuffled_idx].reset_index(drop=True) # shuffle and zero-index df
            self.y = y_infos.iloc[shuffled_idx].reset_index(drop=True) # shuffle and zero-index df
        else:
            self.x = x_infos.reset_index(drop=True) # zero-index df
            self.y = y_infos.reset_index(drop=True) # zero-index df
            
    def __len__(self):
        '''
        Return number of batches, based on batch_size
        '''
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        idx: batch index
        '''
        # print(f'NiftiSequence: getting item {idx}')
        
        # loc indexing uses inclusive name-based indexing, I know I know don't ask, hence the -1.
        batch_x_paths = list(self.x.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'pp_path'])
        # add channel dimension to each augmented (randomly cropped) image.
        batch_x = [np.expand_dims(augment_image(preprocessing.get_preprocessed_image(path)), axis=-1)
                   for path in batch_x_paths]

        batch_y = self.y.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'class']

        # return x and y batches
        return (np.array(batch_x), np.array(batch_y))
    
    def on_epoch_end(self):
        if self.shuffle:
            shuffled_idx = random.sample(range(len(self.x)), k=len(self.x))
            self.x = self.x.iloc[shuffled_idx].reset_index(drop=True) # shuffle and zero-index df
            self.y = self.y.iloc[shuffled_idx].reset_index(drop=True) # shuffle and zero-index df


def get_nifti_datagens(preprocessed_metadata_path, batch_size, crop_shape, seed=0, validation_split=0.25):
    '''
    Return a tuple of training ScanSequence and validation ScanSequence.
    '''
    # Data generator
    infos = preprocessing.read_preprocessed_metadata(preprocessed_metadata_path)
    print('Data set size:', len(infos))
    shuffled = infos.sample(frac=1, random_state=seed)
    nrow = len(shuffled)
    idx = int(nrow * validation_split)
    val = shuffled.iloc[:idx, :].reindex()
    train = shuffled.iloc[idx:, :].reindex()
    print('Train set size:', len(train))
    print('Validation set size:', len(val))
    print('Batch size:', batch_size)
    train_gen = NiftiSequence(train, batch_size, crop_shape)
    val_gen = NiftiSequence(val, batch_size, crop_shape)
    print('Num train batches:', len(train_gen))
    print('Num val batches:', len(val_gen))
    return train_gen, val_gen


def get_nifti_fracture_datagens(preprocessed_metadata_path, batch_size=1, seed=None, validation_split=0.25):
    '''
    Nifti scans are divided into fracture scans and normal scans.  Each group is split into train and validation 
    according to validation_split.  Fracture training and normal training splits are combined into one training
    FractureSequence.  Likewise for validation.
    
    Return a tuple of training FractureSequence and validation FractureSequence
    '''
    assert validation_split <= 1.0 and validation_split >= 0.0
    
    # Data generator
    infos = preprocessing.read_preprocessed_metadata(preprocessed_metadata_path)
    fracture_infos = infos[infos['class'] == 'fracture']
    normal_infos = infos[infos['class'] == 'normal']
    print('Total number of scans:', len(infos))
    print('Number of fracture scans:', len(fracture_infos))
    print('% of total - fracture scans:', len(fracture_infos) / len(infos))
    print('Number of normal scans:', len(normal_infos))
    print('% of total - normal scans:', len(normal_infos) / len(infos))
    
    fracture_shuffled = fracture_infos.sample(frac=1, random_state=seed)
    normal_shuffled = normal_infos.sample(frac=1, random_state=seed+1)
    
    fracture_idx = int(len(fracture_shuffled) * validation_split) # size of val set
    normal_idx = int(len(normal_shuffled) * validation_split) # size of val set.
    
    fracture_train = fracture_shuffled.iloc[fracture_idx:, :].reindex()
    normal_train = normal_shuffled.iloc[normal_idx:, :].reindex()
    print('Number of fracture train scans:', len(fracture_infos))
    print('Number of normal train scans:', len(normal_infos))
    
    fracture_val = fracture_shuffled.iloc[:fracture_idx, :].reindex()
    normal_val = normal_shuffled.iloc[:normal_idx, :].reindex()
    print('Number of fracture validation scans:', len(fracture_infos))
    print('Number of normal validation scans:', len(normal_infos))
    
    # 1 = fracture and 0 = normal
    train_x = pd.concat([fracture_train, normal_train])
    train_y = pd.DataFrame({'class': [1] * len(fracture_train) + [0] * len(normal_train)})
    
    val_x = pd.concat([fracture_val, normal_val])
    val_y = pd.DataFrame({'class': [1] * len(fracture_val) + [0] * len(normal_val)})
    
    print('Train set size:', len(train_x), len(train_y))
    print('Validation set size:', len(val_x), len(val_y))
    print('Batch size:', batch_size)
    train_gen = FractureSequence(train_x, train_y, batch_size=batch_size)
    val_gen = FractureSequence(val_x, val_y, batch_size=batch_size)
    print('Num train batches:', len(train_gen))
    print('Num val batches:', len(val_gen))
    return train_gen, val_gen

