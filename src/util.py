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


def temp_gif_path(tmp_dir):
    '''
    Used to junk up the filesystem with temporary files for animated gifs of ct scans.
    '''
    return str(Path(tmp_dir) / ('tmp_' + uuid.uuid4().hex + '.gif'))

    
def get_nifti_files(path):
    '''
    Return a list of Path objs for every .nii file within path.
    '''
    return list(path.glob('**/*.nii'))


def sample_stack(stack, rows=3, cols=3, start_with=0, show_every=3, r=0):
    '''
    Plot a grid of images (2d slices) sampled from stack.
    
    stack: 3-d voxel array.
    '''
    fig, ax = plt.subplots(rows, cols, figsize=[20, 20])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[i // cols, i % cols].set_title('slice %d' % ind)
        
        if r == 0:
            ax[i // cols, i % cols].imshow(stack[:, :, ind], cmap='gray')
        else:
            ax[i // cols, i % cols].imshow(rotate(stack[:, :, ind], r), cmap='gray')
        
        
        ax[i // cols, i % cols].axis('off')
    plt.show()


def make_animated_gif(path, img, start=0, stop=None, step=1):
    '''
    Create animated gif of 3d image, where each frame is a 2-d image taken from 
    iterating across the 3rd dimension.  E.g. the ith 2d image is img[:, :, i]
    path: where to save the animated gif
    img: a 3-d volume
    start: index of 3rd dimension to start iterating at.  default = 0.
    stop: index of 3rd dimension to stop at, not inclusive.  Default is None, meaning stop at img.shape[2].
    step: number of slices to skip    
    '''
    # convert to uint8 to suppress warnings from imageio
    imax = img.max()
    imin = img.min()
    img = 255 * ((img - imin) / (imax - imin)) # scale to 0..255
    img = np.uint8(img)
    
    with imageio.get_writer(path, mode='I') as writer:
        for i in range(start, img.shape[2], step):
            writer.append_data(img[:, :, i])

    
def animate_crop(img, crop=(0, 1, 0, 1, 0, 1), axis=2, step=5, tmp_dir='/tmp'):
    '''
    img: a 3d volume to be cropped and animated.
    axis: 0, 1, 2: the axis to animate along.  img will be transposed s.t. this axis is the 3rd axis.
    crop: 6 element list: axis 0 start position, axis 0 end position, axis 1 start position, etc.  Each position 
      is a number in [0.0, 1.0] representing the position as a proportion of that axis.  0.0 is the beginning,
      1.0 the end, and 0.5 the middle.
    step: only include every nth frame in the animation, where each frame is a 2d slice of img.
    return: ipython Image, for display in a notebook.
    '''
    # as a proportion of the total range, range of axis 0, 1, and 2 that should be included in the volume
    prop0 = crop[0:2]
    prop1 = crop[2:4]
    prop2 = crop[4:6]
    # as specific voxel coordinates, range of axis 0, 1, and 2 that should be included in the volume
    pix0 = [int(p * img.shape[0]) for p in prop0]
    pix1 = [int(p * img.shape[1]) for p in prop1]
    pix2 = [int(p * img.shape[2]) for p in prop2]

    cropped_img = img[pix0[0]:pix0[1], pix1[0]:pix1[1], pix2[0]:pix2[1]]
    # rotate axes for animation
    cropped_img = cropped_img.transpose([0,1,2][-(2-axis):] + [0,1,2][:-(2-axis)])
    
    tmp_path = temp_gif_path(tmp_dir=tmp_dir)
    print('temp gif path:', tmp_path)
    make_animated_gif(tmp_path, cropped_img, step=step)
    return Image(filename=tmp_path)


def animate_scan_info_crop(scan_info, i, crop=(0, 1, 0, 1, 0, 1), axis=0, step=3):
    path = scan_info.loc[i, 'path']
    print('scan path:', path)
    img = nib.load(path).get_data()
    print('scan img shape:', img.shape)
    return animate_crop(img, crop, axis=axis, step=step)
    

def get_nifti_infos(paths):
    '''
    paths: paths to nifti scans.
    get file paths, image paths, other useful information about data.
    Can be randomly shuffled and split to for train and test set.
    Generator will split examples into batch sizes and get associated normalized images and labels.  
    '''
    infos = pd.DataFrame({'id': [re.sub('\.nii$', '', p.name) for p in paths], 'path': [str(p) for p in paths]})
    infos['nft'] = infos.path.apply(lambda p: nib.load(p))
    infos['header'] = infos.nft.apply(lambda nft: nft.header)
    infos['affine'] = infos.nft.apply(lambda nft: nft.affine)
    infos['pixdim'] = infos.header.apply(lambda h: h['pixdim'][1:4])
    infos['dim'] = infos.header.apply(lambda h: h['dim'][1:4])
    infos['qform_code'] = infos.header.apply(lambda h: h['qform_code'])
    infos['sform_code'] = infos.header.apply(lambda h: h['sform_code'])
    infos['sizeof_hdr'] = infos.header.apply(lambda h: h['sizeof_hdr'])
    infos['pixdim0'] = infos.pixdim.apply(lambda x: x[0])
    infos['pixdim1'] = infos.pixdim.apply(lambda x: x[1])
    infos['pixdim2'] = infos.pixdim.apply(lambda x: x[2])
    infos['dim0'] = infos.dim.apply(lambda x: x[0])
    infos['dim1'] = infos.dim.apply(lambda x: x[1])
    infos['dim2'] = infos.dim.apply(lambda x: x[2])
    infos['desc'] = infos.header.apply(lambda h: h['descrip'])
    infos.reset_index(drop=True)
    return infos



def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
