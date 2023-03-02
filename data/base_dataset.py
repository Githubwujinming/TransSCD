"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image,ImageFilter
import torchvision.transforms as transforms
import albumentations.augmentations.functional as F
from abc import ABC, abstractmethod
import math

import albumentations as A

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_albumentations(opt, grayscale=False,blur=True, convert=True,params=None,noise=True, normalize=True,hsvshit=True, method=cv2.INTER_NEAREST,test=False):
    transform_list = []
    if grayscale:
        transform_list.append(A.ToGray())
    if 'resize' in opt.preprocess:
        transform_list.append(A.Resize(opt.load_size, opt.load_size,interpolation=method))
    if 'blur' in opt.preprocess and test==False:
        transform_list.append(A.GaussianBlur(p=0.3))
    if 'rotate' in opt.preprocess and test==False :
        transform_list.append(A.Rotate(limit=opt.angle, p=0.3))
    if 'transpose' in opt.preprocess and test==False :#同rotate
        transform_list.append(A.Transpose(p=0.2))
    if 'hsvshift' in opt.preprocess and test==False:
        transform_list.append(A.HueSaturationValue(10,5,10,p=0.3))
    if 'noise' in opt.preprocess  and test==False:
        transform_list.append(A.GaussNoise(var_limit=(10,40),p=0.3))
    # if 'crop' in opt.preprocess:
    #     transform_list.append(A.Crop(p=0.4))
    if 'flip' in opt.preprocess  and test==False:#同rotate
        transform_list.append(A.Flip(p=0.3))
    if normalize:
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    if convert:
        transform_list.append(ToTensorV2())
    return A.Compose(transform_list,
                additional_targets={
                    'imageB': 'image',
                    'labelA': 'mask',
                    'labelB': 'mask',
                    'cd':'mask',
                    })

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

