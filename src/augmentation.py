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

def random_crop_indices(img_shape, crop_shape):
    '''
    Choose a random bounding box inside img_shape of size crop_shape.
    
    returns: a list of (start_index, end_index) for each dimension.
    '''
    
    assert all(img_shape[i] >= crop_shape[i] for i in range(len(crop_shape)))
    
    maxes = [img_shape[i] - crop_shape[i] for i in range(len(crop_shape))]
    # the starting corner of the crop
    starts = [random.randint(0, m) for m in maxes]
    
    bounds = [(starts[i], starts[i] + crop_shape[i]) for i in range(len(crop_shape))]
    return bounds

    
def random_crop(img, shape):
    '''
    Randomly crop an image to a shape.  Location is chosen at random from
    all possible crops of the given shape.
    
    img: a volume to crop
    shape: size of cropped volume.  e.g. (32, 32, 32)
    '''
    assert all(img.shape[i] >= shape[i] for i in range(len(shape)))
    # start and end positions of the crop for each dimension of img.
    bounds = random_crop_indices(img.shape, shape)
    return crop_image(img, bounds)
        

def crop_image(img, bounds):
    # Whoa! Cropping a variable dimension shape. Cool.
    return img[[slice(bound[0], bound[1]) for bound in bounds]]


# def crop_images(imgs, bounds):
#     '''
#     Useful for cropping an image and a mask with the same crop
#     '''
#     return [crop_image(img, bounds) for img in imgs]


## FLIPPING



def random_flip_dimensions(n_dims, p=0.5):
    '''
    Randomly select dimensions to flip with a probability p.
    Return a list of dimensions to be flipped.
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
    '''
    returns: a permutation of the dimensions (used to transpose dimensions in an np array)
    '''
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

def random_augmentations(shape, crop_shape=None, flip=None, transpose=False):
    bounds = augmentation.random_crop_indices(shape, crop_shape) if crop_shape else None
    flip_dims = random_flip_dimensions(len(shape), p=flip) if flip else None
    transpose_dims = random_transpose_dimensions(len(shape)) if transpose else None
    return bounds, flip_dims, transpose_dims


def _augment_image(img, gray_std=None, gray_disco=False, crop_bounds=None, flip_dims=None, transpose_dims=False):
    if crop_bounds is not None:
        img = crop_image(img, bounds)

    if gray_std is not None:
        img = gray_value_augment_image(img, std=gray_std, disco=gray_disco)
    
    if flip_dims:
        img = flip_image(img, flip_dims)
        
    if transpose_dims:
        img = transpose_image(img, transpose_dims)

    return img


def augment_image_and_mask(img, mask, crop_shape=None, gray_std=None, gray_disco=False, flip=None, transpose=False):
    '''
    Augment an image and mask with the same random crop, flips, and transpositions.
    '''
    # image and mask must have the same crop, flip, and transpose to match.
    bounds, flip_dims, transpose_dims = random_augmentations(img.shape, crop_shape=crop_shape, flip=flip, transpose=transpose)
    img = _augment_image(img, gray_std=gray_std, gray_disco=gray_disco, crop_bounds=crop_bounds, 
                         flip_dims=flip_dims, transpose_dims=transpose_dims)
    mask = _augment_image(mask, crop_bounds=crop_bounds, flip_dims=flip_dims, transpose_dims=transpose_dims)
    return img, mask


def augment_image(img, crop_shape=None, gray_std=None, gray_disco=False, flip=None, transpose=False):
    '''
    Augment an image randomly with cropping, gray value scaling, flipping, transposing axes.
    Return augmented image.
    
    gray_disco: a silly argument for debugging which gives each slice a different gray augmentation, which creates
      a strobe-like effect when animating the image by slice.
    '''
    bounds, flip_dims, transpose_dims = random_augmentations(img.shape, crop_shape=crop_shape, flip=flip, transpose=transpose)
    img = _augment_image(img, gray_std=gray_std, gray_disco=gray_disco, crop_bounds=crop_bounds, 
                         flip_dims=flip_dims, transpose_dims=transpose_dims)
    return img





