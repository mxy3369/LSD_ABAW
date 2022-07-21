# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import os
import PIL
import cv2
import matplotlib as plt
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from timm.data import create_transform
from timm.data import transforms as tfs
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class MTL_DataSet(Dataset):
    def __init__(self, lt, dataset_str, img_dir):
        self.lt = lt
        self.dataset_str = dataset_str
        self.img_dir = Path(img_dir)

        if self.dataset_str in ['train']:
            self.preprocess = transforms.Compose([
                tfs.RandomResizedCropAndInterpolation(size=(112, 112), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        elif dataset_str in ['valid', 'test']:
            self.preprocess = transforms.Compose([
                # transforms.Resize(size=256, interpolation=PIL.Image.BICUBIC, max_size=None, antialias=None),
                transforms.Resize(size=112, interpolation=PIL.Image.BICUBIC, max_size=None, antialias=None),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        else:
            print('Not Existing dataset_str type')
            exit(0)
    
    def __getitem__(self, index):
        img_name = self.lt[index][0]
        img_path = self.img_dir / img_name
        X = Image.open(img_path)
        X = self.preprocess(X)
        y = self.lt[index][3]
        y = int(y)

        return X, y

    def __len__(self):

        return len(self.lt)
        
class MultiPie_Dataset(Dataset):
    def __init__(self, dataset_dir, csv_path, dataset_type, resize_shape, sample):
        self.dataset_dir = dataset_dir
        self.csv_path = csv_path
        self.dataset_type = dataset_type
        self.resize_shape = resize_shape
        self.sample = sample
        self.lines = []
        self.split_dataset()
        if self.dataset_type in ['train']:
            self.preprocess = transforms.Compose([
                tfs.RandomResizedCropAndInterpolation(size=(112, 112), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        elif dataset_type in ['valid', 'test']:
            self.preprocess = transforms.Compose([
                # transforms.Resize(size=256, interpolation=PIL.Image.BICUBIC, max_size=None, antialias=None),
                transforms.Resize(size=112, interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        else:
            print('Not Existing dataset_type type')
            exit(0)
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split()
        image = Image.open(os.path.join(self.dataset_dir, line[0])).convert('RGB')
        image = self.preprocess(image)
        label = int(line[1])
        return image, label

    def split_dataset(self):
        with open(self.csv_path) as f:
            lines = f.readlines()
        i = 0
        if self.dataset_type in ['train']:
            for t in self.sample:
                for j in range(55):
                    self.lines.append(lines[t*55+j])
                i += 1
                if i > 129 - int(129/5):
                    break
        if self.dataset_type in ['valid', 'test']:
            for t in self.sample:
                if i <= 129 - int(129/5):
                    i += 1
                    continue
                for j in range(55):
                    self.lines.append(lines[t*55+j])
        print(len(self.lines))

class S_Dataset(Dataset):
    def __init__(self, dataset_dir, dataset_type, emotion=None, csv_path=''):
        # self.hand_train_list = [(1917, 1943), (2976, 3086), (3242, 3248), (4769, 4864), (5035, 5163), (5174, 5688), (5816, 5829),\
        #                         (5848, 5880), (7409, 7533), (7892, 7911), (8022, 8391), (8421, 8525), (8559, 8560), (8688, 8917),\
        #                         (8983, 9250), (9492, 9498), (9911, 9912), (10829, 10850)]
        self.hand_train_list = [(7409, 7533), (7892, 7911), (8022, 8391)]
        self.hand_val_list = [(89, 100), (116, 353), (364, 421), (424, 426), (428, 479)]
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.csv_path = csv_path
        self.cls2lab = {'ANGRER':0, 'DISGUST':1, 'FEAR':2, 'HAPPINESS':3, 'SADNESS':4, 'SURPRISE':5}
        self.lab2cls = {0:'ANGRER', 1:'DISGUST', 2:'FEAR', 3:'HAPPINESS', 4:'SADNESS', 5:'SURPRISE'}
        self.emotion = [self.lab2cls[i] for i in range(6)] if emotion is None else emotion
        self.hand_train_label, self.hand_test_label = self.fear_hand_label()
        self.data = []
        if self.dataset_type not in ['test', 'test_hand']:
            self.get_images()
        else:
            self.get_lines()
        
        
        if self.dataset_type in ['train']:
            self.preprocess = transforms.Compose([
                tfs.RandomResizedCropAndInterpolation(size=(112, 112), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        elif dataset_type in ['valid', 'test', 'test_hand']:
            self.preprocess = transforms.Compose([
                # transforms.Resize(size=256, interpolation=PIL.Image.BICUBIC, max_size=None, antialias=None),
                transforms.Resize(size=112, interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        elif dataset_type in ['hand', 'hand_train', 'hand_test']:
            self.preprocess = transforms.Compose([
                # transforms.Resize(size=256, interpolation=PIL.Image.BICUBIC, max_size=None, antialias=None),
                transforms.Resize(size=112, interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        else:
            print('Not Existing dataset_type type')
            exit(0)
    
    def __getitem__(self, idx):
        if self.dataset_type in ['test', 'test_hand']:
            image_data = self.data[idx].strip().split()
            image = cv2.imread(os.path.join(self.dataset_dir, image_data[0]))
            if self.dataset_type in ['test_hand']:
                grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = self.Sobel(grayImage)
            # label = int(image_data[1])
            image = Image.fromarray(image).convert('RGB')
            image = self.preprocess(image)
            return image, 0, image_data[0]
        if self.dataset_type == 'hand_train':
            idx = (idx % 982 + 7409)
        image_data = self.data[idx]
        image = cv2.imread(os.path.join(self.dataset_dir, self.lab2cls[image_data[1]], image_data[0]))
        if self.dataset_type in ['hand', 'hand_train', 'hand_test']:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = self.Sobel(grayImage)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # return image
        image = Image.fromarray(image).convert('RGB')
        image = self.preprocess(image)
        if self.dataset_type == 'hand_train':
            label = int(self.hand_train_label[idx])
        elif self.dataset_type == 'hand_test':
            label = int(self.hand_test_label[idx])
        else:
            label = int(image_data[1])
        return image, label, os.path.join(self.lab2cls[image_data[1]], image_data[0])
    
    def __len__(self):
        # print(len(self.data))
        return len(self.data)

    def Sobel(self, grayImage):
        # 灰度化处理图像
        # Sobel算子
        x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  # 对x求一阶导
        y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)  # 对y求一阶导
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return Sobel
 
    def get_images(self):
        # sum_F = 0
        for emo in sorted(os.listdir(self.dataset_dir)):
            if emo == 'processed':
                emo = 'FEAR'
                # continue
            if emo in self.emotion:
                for image_name in sorted(os.listdir(os.path.join(self.dataset_dir, emo))):
                    if image_name[-3:] == 'jpg':
                        self.data.append([image_name, self.cls2lab[emo]])
                        # sum_F += 1
    def get_lines(self):
         with open(self.csv_path) as f:
            lines = f.readlines()
            self.data = lines
    
    def fear_hand_label(self):
        sum_train = 0
        l_train = np.zeros(10923)
        l_val = np.zeros(523)
        for a, b in self.hand_train_list:
            l_train[a:b] = 1
            sum_train += b-a
        for a, b in self.hand_val_list:
            l_val[a:b] = 1

        return l_train, l_val

