import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import util.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
# from osgeo import gdal_array
import cv2
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset

num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

root = '/data/datasets/SECOND_scd/'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs

def read_RSimages(mode, rescale=False):
    #assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_A_dir = os.path.join(root, mode, 'A_L')
    label_B_dir = os.path.join(root, mode, 'B_L')
    # To use rgb labels:
    #label_A_dir = os.path.join(root, mode, 'label1_rgb')
    #label_B_dir = os.path.join(root, mode, 'label2_rgb')
    
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            
            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            
            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)
            #for rgb labels:
            label_A = Color2Index(label_A)
            label_B = Color2Index(label_B)
            labels_A.append(label_A)
            labels_B.append(label_B)
        count+=1
        if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))
    
    # print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgs_list_A, imgs_list_B, labels_A, labels_B

class RSSTDataset(BaseDataset):
    def __init__(self, opt):
        self.random_flip = False
        folder_A = 'A'
        folder_B = 'B'
        folder_AL = 'A_L'
        folder_BL = 'B_L'
        self.istest = False
        if opt.phase == 'test':
            self.istest = True
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages('train')
        if opt.phase == 'val':
            self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages('test')

    
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        CD_L = (label_A>0).astype(np.uint8)
        if self.random_flip:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B)
        return {'A': F.to_tensor(img_A).float(), 'A_paths': self.imgs_list_A[idx],
                'B': F.to_tensor(img_B).float(), 'B_paths': self.imgs_list_B[idx],
                'A_L':F.to_tensor(label_A).squeeze().long(), 'AL_path':self.labels_A[idx],
                'B_L':F.to_tensor(label_B).squeeze().long(), 'BL_path':self.labels_B[idx],
                'CD_L':F.to_tensor(CD_L).long().squeeze()
                }
        # return F.to_tensor(img_A), F.to_tensor(img_B), F.to_tensor(label_A), F.to_tensor(label_B), F.to_tensor(CD_L)

    def __len__(self):
        return len(self.imgs_list_A)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'im1')
        imgB_dir = os.path.join(test_dir, 'im2')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len

