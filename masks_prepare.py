# -*- coding: utf-8 -*-
"""

@author: serdarhelli
"""

import os
import numpy as np
from PIL import Image
from zipfile import ZipFile
from natsort import natsorted
from pathlib import Path

# Use __file__ for reliable path resolution (works in scripts and notebooks)
_THIS_DIR = Path(__file__).parent.resolve()
_ORIGINAL_MASKS_DIR = _THIS_DIR / 'Original_Masks'

def convert_one_channel(img):
    #some images have 3 channels , although they are grayscale image
    if len(img.shape)>2:
        img=img[:,:,0]
        return img
    else:
        return img
def pre_masks(resize_shape=(512,512), path=None):
    if path is None:
        path = _ORIGINAL_MASKS_DIR
    else:
        path = Path(path)
    ZipFile(path / "Orig_Masks.zip").extractall(path / 'Masks')
    masks_path = path / 'Masks'
    dirs = natsorted(os.listdir(masks_path))
    masks = img = Image.open(masks_path / dirs[0])
    masks = masks.resize(resize_shape, Image.LANCZOS)
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(masks_path / dirs[i])
        img = img.resize(resize_shape, Image.LANCZOS)
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), resize_shape[0], resize_shape[1], 1))
    return masks


_CUSTOM_MASKS_DIR = _THIS_DIR / 'Custom_Masks'


# CustomMasks 512x512
def pre_splitted_masks(path=None):
    if path is None:
        path = _CUSTOM_MASKS_DIR
    else:
        path = Path(path)
    ZipFile(path / "splitted_masks.zip").extractall(path / 'Masks')
    masks_path = path / 'Masks'
    dirs = natsorted(os.listdir(masks_path))
    masks = img = Image.open(masks_path / dirs[0])
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(masks_path / dirs[i])
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), 512, 512, 1))
    return masks
    




    
