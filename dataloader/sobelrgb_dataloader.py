import os
import random
import numpy as np
import cv2

import torch
from PIL import Image
import torchvision
import torchvision.transforms.functional as F

def Sobel(image):
    #sob = image.copy()
    sob = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY) 
    sobelX = cv2.convertScaleAbs( cv2.Sobel(sob, cv2.CV_64F, 1, 0) )
    sobelY = cv2.convertScaleAbs( cv2.Sobel(sob, cv2.CV_64F, 0, 1) )
    sob = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    return sob

class CrowdDataset(torch.utils.data.Dataset):

    def __init__(self, labeled_file_list, labeled_main_transform=None, labeled_img_transform=None, labeled_dmap_transform=None):

        self.labeled_data_files = []
        with open(labeled_file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.labeled_data_files.append(line.strip())
        f.close()

        self.label_main_transform = labeled_main_transform
        self.label_img_transform = labeled_img_transform
        self.label_dmap_transform = labeled_dmap_transform

    def __len__(self):
        return len(self.labeled_data_files)

    def __getitem__(self, index):
        index = index % len(self.labeled_data_files)
        labeled_image_filename = self.labeled_data_files[index]
        labeled_gt_filename = labeled_image_filename.replace('Train', 'Train_gt').replace('Test', 'Test_gt').replace('.jpg', '.npy')

        img = Image.open(labeled_image_filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        dmap = np.load(labeled_gt_filename, allow_pickle=True)
        dmap = dmap.astype(np.float32, copy=False)
        #dmap = Image.fromarray(dmap)
        
        sob = np.array(Sobel(img))
        img = np.array(img)

        h, w = sob.shape
        img4 = np.empty((h, w, 4))
        for i in range(h):
            for j in range(w):
                img4[i][j] = np.append(img[i][j],[sob[i][j]])
        img = img4
        
        if self.label_main_transform is not None:
            img, dmap = self.label_main_transform((img, dmap))
        if self.label_img_transform is not None:
            img = self.label_img_transform(img)
        if self.label_dmap_transform is not None:
            dmap = self.label_dmap_transform(dmap)

        return {'image': img, 'densitymap': dmap, 'imagepath': labeled_image_filename}


def get_train_shanghaitechpartA_dataloader(labeled_file_list, use_flip, batch_size=1, mean=[0.5,0.5,0.5,0.5], std=[0.225,0.225,0.225,0.225]):
    main_transform_list = []

    main_transform_list.append(PairedCrop())

    main_transform = torchvision.transforms.Compose(main_transform_list)
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    densitymap_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    dataset = CrowdDataset(
        labeled_file_list=labeled_file_list,
        labeled_main_transform=main_transform,
        labeled_img_transform=image_transform,
        labeled_dmap_transform=densitymap_transform
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


def get_test_shanghaitechpartA_dataloader(file_list):
    main_transform_list = []
    main_transform_list.append(PairedCrop())
    main_transform = torchvision.transforms.Compose(main_transform_list)
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225, 0.225])
    ])
    densitymap_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    dataset = CrowdDataset(
        labeled_file_list=file_list,
        labeled_main_transform=main_transform,
        labeled_img_transform=image_transform,
        labeled_dmap_transform=densitymap_transform
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return dataloader

class PairedCrop:
    '''
    Paired Crop for both image and its density map.
    Note that due to the maxpooling in the nerual network,
    we must promise that the size of input image is the corresponding factor.
    '''

    def __init__(self, factor=8): # since the CSRNet uses Maxpooling three times in the frontend layers.
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        #w, h = img.size
        h, w = img.shape[0], img.shape[1]
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, data):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = data

        i, j, th, tw = self.get_params(img, self.factor)

        #img = F.crop(img, i, j, th, tw)
        #dmap = F.crop(dmap, i, j, th, tw)

        img = img[i:i+th, j:j+tw, :]
        dmap = dmap[i:i+th, j:j+tw]
        return (img, dmap)

