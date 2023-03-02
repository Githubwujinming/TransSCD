import torch
import random
from re import L
from data.base_dataset import BaseDataset, get_albumentations
from data.image_folder import make_dataset
import os
import cv2
import numpy as np

num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

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

def TensorIndex2Color(pred):
    colormap = torch.as_tensor(ST_COLORMAP, dtype=torch.uint8)
    x = pred.long()
    return colormap[x, :]
class SECONDDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    datafolder-tree
    dataroot:.
            ├─A
            ├─B
            ├─A_L
            ├─B_L
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        folder_A = 'A'
        folder_B = 'B'
        folder_AL = 'A_L'
        folder_BL = 'B_L'
        self.istest = False
        if opt.phase == 'test':
            self.istest = True
        self.A_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_A), opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_B), opt.max_dataset_size))
        if not self.istest:
            self.AL_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_AL), opt.max_dataset_size))
            self.BL_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_BL), opt.max_dataset_size))

        # print(self.A_paths)

    def _get_item_alb(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = cv2.cvtColor(cv2.imread(A_path),cv2.COLOR_BGR2RGB)
        B_img = cv2.cvtColor(cv2.imread(B_path),cv2.COLOR_BGR2RGB)
        AL_path = self.AL_paths[index]
        BL_path = self.BL_paths[index]
        #label 转成对应的索引
        AL_img = cv2.cvtColor(cv2.imread(AL_path),cv2.COLOR_BGR2RGB)
        AL_img = Color2Index(AL_img)
        BL_img = cv2.cvtColor(cv2.imread(BL_path),cv2.COLOR_BGR2RGB)
        BL_img = Color2Index(BL_img)
        CD_L = (AL_img>0).astype(np.uint8)

        transform = get_albumentations(self.opt, test=self.istest)
        transformed = transform(image=A_img,imageB=B_img,labelA=AL_img,labelB=BL_img,cd=CD_L)
        if self.istest:
            return {'A': transformed['image'], 'A_paths': A_path, 'B': transformed['imageB'], 'B_paths': B_path}
        return {'A': transformed['image'], 'A_paths': A_path,
                'B': transformed['imageB'], 'B_paths': B_path,
                'A_L':transformed['labelA'], 'AL_path':AL_path,
                'B_L':transformed['labelB'], 'BL_path':BL_path,
                'CD_L':transformed['cd']
                }
    def __getitem__(self, index):
        # return self._get_item_ori(index)
        return self._get_item_alb(index)
# 
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
