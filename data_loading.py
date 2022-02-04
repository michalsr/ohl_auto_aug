from operator import eq
from turtle import pos
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle
from data_transforms import *
from torch.utils.data import random_split
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from misc import  Cutout
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.transforms as transforms 
from torch.utils.data.sampler import SubsetRandomSampler
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10(data.Dataset):
    folder = ''
    def __init__(self,train=True,augmentations=None,test=False,auto_aug=True):
        self.train = train
        self.test = test
        self.auto_aug = auto_aug
        self.augmentations = augmentations
        self.augmentation_list = self.make_aug_dict()
        self.data = []
        self.labels = []
        
        #take 
        if not test:
            for i in range(1,6):
                with open(f'/home/michal/ohl_auto_aug/cifar-10-batches-py/data_batch_{i}', 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.labels.extend(entry['labels'])
        else:
            with open(f'/home/michal/ohl_auto_aug/cifar-10-batches-py/test_batch', 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    
    def get_samplers(self,valid_size=0.1):
        num_train = len(self.data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        return train_sampler,valid_sampler,train_idx
    def apply_transform(self,img,img_augs):
        #apply previous transforms 
        #apply img augs
        #apply cutout
        MEAN, STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        t=  transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        img_augs[0],img_augs[1],
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(MEAN, STD),
    ])
        return t(img) 
    def regular_transform(self,img):
        #apply previous transforms 
        #apply img augs
        #apply cutout
        MEAN, STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        t=  transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
        return t(img) 
    def normal_transform(self,img):
        MEAN, STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        t=  transforms.Compose([
        transforms.ToTensor(),   
        transforms.Normalize(MEAN, STD),
    ])
        return t(img) 
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

  
        if self.train==True:
            if self.auto_aug:
                #print('Loading auto aug')
                augs = self.augmentations[index]
                
                img_augs = self.get_custom_augs(augs)
                img = self.apply_transform(img,img_augs)
            else:
                #print('Loading normal train transforms')
                img = self.regular_transform(img)
        else:
            #print('Loading normal for val and test')
            img = self.normal_transform(img)
  
        return img, target
    def get_custom_augs(self,augs):
        aug_1,aug_2 = self.augmentation_list[augs]
       
        return aug_1,aug_2
    def make_aug_dict(self):
   
        aug_dict={0:ShearX(0.1),1:ShearX(0.2),2:ShearX(0.3),3:ShearY(0.1),4:ShearY(0.2),5:ShearY(0.3),6:TranslateX(0.15),
        7:TranslateX(0.3),8:TranslateX(0.45),9:TranslateY(0.15),10:TranslateY(0.3),11:TranslateY(0.45),12:Rotate(10),
        13:Rotate(20),14:Rotate(30),15:Color(0.3),16:Color(0.6),17:Color(0.9),18:Posterize(4),19:Posterize(5),20:Posterize(8),21:Solarize(26),22:Solarize(102),23:Solarize(179),
        24:Contrast(1.3),25:Contrast(1.6),26:Contrast(1.9),27:Sharpness(1.3),28:Sharpness(1.6),29:Sharpness(1.9),30:Brightness(1.3),31:Brightness(1.6),32:Brightness(1.9),
        33:AutoContrast(),34:Equalize(),35:Invert()}
        index_start = 0
        final_dict = {}
        for k_1 in aug_dict.keys():
            for k_2 in aug_dict.keys():
                final_dict[index_start] = (aug_dict[k_1],aug_dict[k_2])
                index_start += 1 
        return final_dict 

                


class CIFAR10Test(data.Dataset):
    folder = ''
    def __init__(self,train=True,augmentations=None,test=False,auto_aug=True):
        self.train = train
        self.test = test
        self.auto_aug = auto_aug
        self.augmentations = augmentations
        self.augmentation_list = self.make_aug_dict()
        self.data = []
        self.labels = []
        
        #take 
        if not test:
            for i in range(1,6):
                with open(f'/home/michal/ohl_auto_aug/cifar-10-batches-py/data_batch_{i}', 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.labels.extend(entry['labels'])
        else:
            with open(f'/home/michal/ohl_auto_aug/cifar-10-batches-py/test_batch', 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    
    def get_samplers(self,valid_size=0.1):
        num_train = len(self.data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        return train_sampler,valid_sampler,train_idx
    def apply_transform(self,img,img_augs):
        #apply previous transforms 
        #apply img augs
        #apply cutout
        MEAN, STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        t=  transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        img_augs[0],img_augs[1],
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(MEAN, STD),
    ])
        return t(img) 
    def regular_transform(self,img):
        #apply previous transforms 
        #apply img augs
        #apply cutout
        MEAN, STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        t=  transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
        return t(img) 
    def normal_transform(self,img):
        MEAN, STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        t=  transforms.Compose([
        transforms.ToTensor(),   
        transforms.Normalize(MEAN, STD),
    ])
        return t(img) 
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

  
        if self.train==True:
            if self.auto_aug:
                #print('Loading auto aug')
                augs = self.augmentations[index]
                
                img_augs = self.get_custom_augs(augs)
                img = self.apply_transform(img,img_augs)
            else:
                #print('Loading normal train transforms')
                img = self.regular_transform(img)
        else:
            #print('Loading normal for val and test')
            img = self.normal_transform(img)
  
        return img, target
    def get_custom_augs(self,augs):
        aug_1,aug_2 = self.augmentation_list[augs]
       
        return aug_1,aug_2
    def make_aug_dict(self):
   
        aug_dict={0:ShearX(0.1),1:ShearX(0.2),2:ShearX(0.3),3:ShearY(0.1),4:ShearY(0.2),5:ShearY(0.3),6:TranslateX(0.15),
        7:TranslateX(0.3),8:TranslateX(0.45),9:TranslateY(0.15),10:TranslateY(0.3),11:TranslateY(0.45),12:Rotate(10),
        13:Rotate(20),14:Rotate(30),15:Color(0.3),16:Color(0.6),17:Color(0.9),18:Posterize(4),19:Posterize(5),20:Posterize(8),21:Solarize(26),22:Solarize(102),23:Solarize(179),
        24:Contrast(1.3),25:Contrast(1.6),26:Contrast(1.9),27:Sharpness(1.3),28:Sharpness(1.6),29:Sharpness(1.9),30:Brightness(1.3),31:Brightness(1.6),32:Brightness(1.9),
        33:AutoContrast(),34:Equalize(),35:Invert()}
        index_start = 0
        final_dict = {}
        for k_1 in aug_dict.keys():
            for k_2 in aug_dict.keys():
                final_dict[index_start] = (aug_dict[k_1],aug_dict[k_2])
                index_start += 1 
        return final_dict 

                

