'''
Utilities for building models.  For training models.  For evaluating and visualizing models.
'''

import datetime
import importlib
import keras
from keras.layers import (Dense, SimpleRNN, Input, Conv1D, 
                          LSTM, GRU, AveragePooling3D, MaxPooling3D, Conv3D, 
                          UpSampling3D, BatchNormalization, Concatenate)
from keras.models import Model
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
import IPython.display
from IPython.display import Image # display animated volumes
from IPython.display import SVG # visualize model
from keras.utils.vis_utils import model_to_dot # visualize model

# for importing local code
src_dir = str(Path(projd.cwd_token_dir('notebooks')) / 'src') # $PROJECT_ROOT/src
if src_dir not in sys.path:
    sys.path.append(src_dir)

import util
importlib.reload(util)
import preprocessing
importlib.reload(preprocessing)
import datagen
importlib.reload(datagen)
        
        
def display(model):
    print(model.summary())
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))


def get_model_path(models_dir, model_name, epoch):
    '''
    Paths match the template that the keras checkpoint callback uses to save models.
    The model is saved as model_name_{epoch}
    '''
    model_path = Path(models_dir) / f'{model_name}_{epoch:02d}.h5'
    return model_path


def train_model_epoch(train_gen, val_gen, epoch, models_dir, model_name, epochs=100, batch_size=4,  
                log_dir=None,
                tensorboard_log_dir=None, max_queue_size=10):
    '''
    epoch: used as initial_epoch in training.  Also used to load the corresponding model checkpoint.
    '''
    
    path = get_model_path(models_dir, model_name, epoch)
    model = keras.models.load_model(path)
    return train_model(model=model, train_gen=train_gen, val_gen=val_gen, epochs=epochs, 
                       initial_epoch=epoch, batch_size=batch_size, models_dir=models_dir, 
                       model_name=model_name, log_dir=log_dir, tensorboard_log_dir=tensorboard_log_dir,
                       max_queue_size=max_queue_size)


def train_model(model, train_gen, val_gen, models_dir, 
                model_name, epochs=100, initial_epoch=0, batch_size=None, log_dir=None,
                tensorboard_log_dir=None, max_queue_size=10):
    callbacks = []
    
    # Saving model
    model_path_template = models_dir / (model_name +'_{epoch:02d}.h5')
    print('model_path_template:', model_path_template)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        str(model_path_template), monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, 
        mode='auto', period=1)
    callbacks.append(checkpoint_cb)
    
    # Stop when validation loss stops improving
    early_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    # callbacks.append(early_cb)

    if log_dir:
        # Save logs for each run to logfile
        log_path = log_dir / (model_name + '_' + datetime.datetime.now().isoformat() + '_log.csv')
        print('log_path:', log_path)
        log_cb = keras.callbacks.CSVLogger(str(log_path), separator=',', append=False)
        callbacks.append(log_cb)
        
    if tensorboard_log_dir:
        # Enable Tensorboard Callback
        print('tensorboard log dir:', tensorboard_log_dir)
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_log_dir), histogram_freq=0, write_graph=True, write_images=True)
        callbacks.append(tensorboard_cb)
    
    # Fit Model
    history = model.fit_generator(train_gen, epochs=epochs, initial_epoch=initial_epoch, validation_data=val_gen, 
                        callbacks=callbacks, max_queue_size=max_queue_size)
    return history, log_path


def vizualize_predictions_by_epochs(models_dir, model_name, epochs, train_gen, val_gen, step=5):
    '''
    Visualize some of the training data and validation data.  Compare ground truth
    to the output of the model from epoch.
    
    Currently only does one batch.  Would love to get a sweet grid of animated gifs happening, but not today.
    
    Yields items to display b/c I can't figure out jupyter notebook display right now.
    '''
    kind_gens = (('train', train_gen), ('val', val_gen))
    kind_batches = [(kind, gen[0]) for kind, gen in kind_gens]
    for epoch in epochs:
        print('Epoch {}:'.format(epoch))
        path = get_model_path(models_dir, model_name, epoch)
        model = keras.models.load_model(path)
        for kind, (batch_x, batch_y) in kind_batches:
            print(f'{kind} data set')
            batch_pred = model.predict_on_batch(batch_x)
            for i in range(len(batch_x)):
                print(f'predicted vs truth for image {i}')
                yield util.animate_crop(np.squeeze(batch_pred[i]), step=step)
                yield util.animate_crop(np.squeeze(batch_y[i]), step=step)
                


def build_residual_encoder_decoder_block(x, n_a, n_d=1):
    '''
    Convolves x,  residually encodes and decodes (like u-net) and convolves to x's original number of channels.
    n_d: the "depth" of the encoder, the number of times the input is downsampled.  
      0 returns a 1-layer convolution of x.
      n returns a returns a u-net-like convolution of x, where x is encoded/downsampled n times
        and the decoder combines then encoded input (or a convolution of it) and 
        the upsampled residual by convolution of concatenated channels.
    n_a: the activation output size, in number of channels, of all convolutions, except when matching the shape of x.
    
    u-net uses 2 convolutions of input per depth, and concatenates input with upsample when decoding.
    refinenet and "large kernel matters" convolve the original input again before combining it 
      with the upsampled residual.
    refinenet and "large kernel matters" convovle before combining, use sum residual combination, 
      and convolve after combining.
    u-net convovles after combining and before downsampling.
      
    A u-net would be similar to n_r=1, n_d=3
    
    Some architectures use 1x1 convolutions, like PSPNet.  Densenet uses 1x1 to compress big stacks of channels
      before a 3x3 convolution.
    Other architectures use dilated convolutions.
    Some architectures use strided and transposed convolutions for downsampling and upsampling.
    Some some/most/all use max pooling over avg pooling.  Don't know why.  Try it.
    '''
    
    x = Conv3D(n_a, kernel_size=(3, 3, 3), padding='same', activation='relu')(x) # (32, 32, 32, 16)

    if n_d > 0:
        x_e = x # shape: (32, 32, 32, 16)
        x_e = MaxPooling3D(padding='same')(x_e) # shape: (16, 16, 16, 16)
        x_e = build_residual_encoder_decoder_block(x_e, n_a, n_d - 1) # recursive call
        x_d = UpSampling3D()(x_e) # shape (32, 32, 32, 16)
        x = Concatenate()([x, x_d]) # residual join.  shape (32, 32, 32, 32)
        x = Conv3D(n_a, kernel_size=(3, 3, 3), padding='same', activation='relu')(x) # (32, 32, 32, 16)
    
    return x

    









