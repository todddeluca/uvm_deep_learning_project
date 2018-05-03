'''
Sources include:
- https://github.com/ellisdg/3DUnetCNN/blob/master/unet3d/augment.py
'''

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


# GREYSCALE AUGMENATION

def random_gray_factor(mean=1, std=0.25):
    return 


def scale_gray_image(image, factor):
    '''
    Whiten (or darken) an image.
    
    image: np.array
    factor: will be multiplied with every value in image, shifting the mean brightness
    '''
    return image * np.asarray(factor)


def gray_value_augment_image(image, mean=1, std=0.1, disco=False):
    '''
    Multiply image by normally distributed noise with mean and std.
    
    image: 3d image (no channel dimension yet. grayscale.)
    mean: Default 1, so the average number that an image is multiplied by is 1.
    std: standard deviation of normal distribution.
    '''
    if disco:
        # each slice gets its own lightening/darkening.  There really should be an axis argument here.
        rnd_shape = image.shape[-1] 
    else:
        # whole image is multiplied by same factor.
        rnd_shape = 1
    return image * np.random.normal(mean, std, rnd_shape)


## SCALING AUGMENTATION

def scale_image(image, scale_factor):
    
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)


## RANDOM CROPPING

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
    # Whoa! Cropping a variable dimension shape. Cool.
    cropped_img = img[[slice(starts[i], starts[i] + shape[i]) for i in range(len(shape))]]
    return cropped_img
        

## FLIPPING



def random_flip_dimensions(n_dims, p=0.5):
    '''
    Randomly select dimensions to flip with a probability p.
    '''
    # random selection of ones at a rate of p.
    idx = np.random.choice(a=[True, False], size=n_dims, p=[p, 1-p])
    # print('idx:', idx)
    dims = np.array(range(n_dims))[idx] # [idx]
    # print('n_dims:', n_dims, ' p:', p, 'dims.shape:', dims.shape, 'dims:', dims)
    return dims


def flip_image(image, dims):
    for dim in dims:
        image = np.flip(image, axis=dim)
        
    return image


def random_transpose_dimensions(n_dims):
    return np.random.permutation(range(n_dims))


def transpose_image(image, dims):
    '''
    image: square, cube, or hypercube
    dims: reorder the axes of image according to this list
    '''
    if not all(d == image.shape[0] for d in image.shape):
        raise Exception('transpose_image only works on images where all dimensions are equal (hypercubes).' , image.shape)

    return np.transpose(image, dims)


## AUGMENTATION


def augment_image(img, crop_shape=None, gray_std=None, gray_disco=False, flip=None, transpose=False):
    '''
    gray_disco: a silly argument for debugging which gives each slice a different gray augmentation, which creates
      a strobe-like effect when animating the image by slice.
    '''
    if crop_shape is not None:
        # print('random crop.  crop_shape:', crop_shape)
        img = random_crop(img, crop_shape)

    if gray_std is not None:
        # print('gray value augmentation.  gray_std:', gray_std)
        img = gray_value_augment_image(img, std=gray_std, disco=gray_disco)
    
    if flip:
        # print('flip augmentation.  flip:', flip)
        img = flip_image(img, random_flip_dimensions(len(img.shape), p=flip))

    if transpose:
        dims = random_transpose_dimensions(len(img.shape))
        # print('transpose augmentation.  random dims:', dims)
        img = transpose_image(img, dims)
        
    return img


def augment_data(img, mask, affine, grey_std=None, flip=True):
    n_dim = len(truth.shape)
    if scale_deviation:
        scale_factor = random_scale_factor(n_dim, std=scale_deviation)
    else:
        scale_factor = None
    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        flip_axis = None
    data_list = list()
    for data_index in range(data.shape[0]):
        image = get_image(data[data_index], affine)
        data_list.append(resample_to_img(distort_image(image, flip_axis=flip_axis,
                                                       scale_factor=scale_factor), image,
                                         interpolation="continuous").get_data())
    data = np.asarray(data_list)
    truth_image = get_image(truth, affine)
    truth_data = resample_to_img(distort_image(truth_image, flip_axis=flip_axis, scale_factor=scale_factor),
                                 truth_image, interpolation="nearest").get_data()
    return data, truth_data



