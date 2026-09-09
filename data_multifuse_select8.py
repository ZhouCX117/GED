import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn as nn
from pathlib import Path
from functools import partial
from utils import exists, convert_image_to_fn, normalize_to_neg_one_to_one
from PIL import Image, ImageDraw
import torch.nn.functional as F
import math
import random
import torchvision.transforms.functional as F2
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Any, Callable, Optional, Tuple
import os
import pickle
import numpy as np
import copy
import albumentations
from torchvision.transforms.functional import InterpolationMode
from glob import glob

class EdgeDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        augment_horizontal_flip = True,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='train',
        threshold=0.3, use_uncertainty=False, cfg={}
    ):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size

        self.threshold = threshold * 255
        self.use_uncertainty = use_uncertainty
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one

        self.data_list = open(os.path.join(data_root,'train_multiple_no01.txt'),"r").readlines()

        crop_type = cfg.get('crop_type') if 'crop_type' in cfg else 'rand_crop'
        if crop_type == 'rand_crop':
            self.transform = Compose([
                RandomCrop(image_size),
                RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
                ToTensor()
            ])
        elif crop_type == 'rand_resize_crop':
            self.transform = Compose([
                RandomResizeCrop(image_size),
                RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
                ToTensor()
            ])
        print("crop_type:", crop_type)


    def __len__(self):
        return len(self.data_list)
    
    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        raw_width, raw_height = img.size
        return img, (raw_width, raw_height) 
   
    
    def read_label(self, img_edge_path):
        label_list,edge_list=[],[]

        for index in range(1,len(img_edge_path)):
            lb_data = Image.open(os.path.join(self.data_root,img_edge_path[index])).convert('L')
            lb = np.array(lb_data).astype(np.float32)
            lb[lb>=0.3*255]=255
            label_list.append(Image.fromarray(lb.astype(np.uint8)))
            
            
            edge_list.append(np.sum(lb/255))
        max_num=edge_list.index(max(edge_list))
        min_num=edge_list.index(min(edge_list))
        max_min=max(edge_list)-min(edge_list)
        gran_list=[]
        for edge_num in edge_list:
            gran=(edge_num-min(edge_list))/max_min
            assert gran>=0 and gran<=1
            gran_list.append(gran)
        return label_list,gran_list,max_num,min_num


    def __getitem__(self, index):
        img_edge_path = self.data_list[index].strip("\n").split("\t")
        img_path=img_edge_path[0]

        img, raw_size = self.read_img(os.path.join(self.data_root,img_path))
        img_name = os.path.basename(img_path)
        label_list,gran_list,max_num_index,min_num_index=self.read_label(img_edge_path)
        
        select_index=torch.randperm(len(gran_list))[:4]
        select_label_list,select_gran_list=[],[]
        for i in select_index:
            select_label_list.append(label_list[i])
            select_gran_list.append(gran_list[i])
        
        select_label_list.append(label_list[max_num_index])
        select_label_list.append(label_list[min_num_index])
        
        img,select_label_list=self.transform(img,select_label_list)
        select_label_torch=torch.cat(select_label_list[:4],0)
        
        max_torch=select_label_list[4]
        min_torch=select_label_list[5]
        min_torch_label=min_torch.clone()
        max_torch=torch.where(max_torch<0.3,0,max_torch)
        max_torch=torch.where(max_torch>=0.3,1,max_torch)
        max_num=torch.sum(max_torch)
        
        min_torch=torch.where(min_torch<0.3,0,min_torch)
        min_torch=torch.where(min_torch>=0.3,1,min_torch)
        min_num=torch.sum(min_torch)
        
        if self.normalize_to_neg_one_to_one:   # transform to [-1, 1]
            img=normalize_to_neg_one_to_one(img)
            select_label_torch = normalize_to_neg_one_to_one(select_label_torch)
            
           
        return {'image': select_label_torch,'gran':select_gran_list, 'cond': img, 'raw_size': raw_size, 'img_name': img_name,'max_num':max_num,'min_num':min_num,'min_torch':min_torch_label}

class EdgeDatasetTest(data.Dataset):
    def __init__(
        self,
        data_root,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
    ):
        super().__init__()

        self.data_root = data_root
        self.image_size = image_size
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one


        self.data_list = self.build_list()

        self.transform = Compose([
            ToTensor()
        ])

    def __len__(self):
        return len(self.data_list)

    def build_list(self):
        data_root = os.path.abspath(self.data_root)
        # images_path = os.path.join(data_root)
        images_path = data_root
        samples = get_imgs_list(images_path)
        return samples
    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        raw_width, raw_height = img.size


        return img, (raw_width, raw_height)

    def read_lb(self, lb_path):
        lb_data = Image.open(lb_path).convert('L')
        lb = np.array(lb_data).astype(np.float32)

        threshold = self.threshold


        lb[lb >= threshold] = 255
        lb = Image.fromarray(lb.astype(np.uint8))
        return lb


    def __getitem__(self, index):
        img_path = self.data_list[index]
        img_name = os.path.basename(img_path)

        img, raw_size = self.read_img(img_path)

        img = self.transform(img)
        if self.normalize_to_neg_one_to_one:   # transform to [-1, 1]
            img = normalize_to_neg_one_to_one(img)
        return {'cond': img, 'raw_size': raw_size, 'img_name': img_name}


def get_imgs_list(imgs_dir):
    imgs_list = os.listdir(imgs_dir)
    imgs_list.sort()
    return [os.path.join(imgs_dir, f) for f in imgs_list if f.endswith('.jpg') or f.endswith('.JPG')or f.endswith('.png') or f.endswith('.pgm') or f.endswith('.ppm')]


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, target=None):
        if target is None:
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
            return img
        elif torch.is_tensor(target):
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
                target = F2.hflip(target)
            return img, target
        elif isinstance(target,list):
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
                return_list=[]
                for item in target:
                    return_list.append(F2.hflip(item))
                return img,return_list
            else:
                return img,target
            

class RandomResizeCrop(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.25, 1.0), **kwargs):
        super().__init__(size, scale, **kwargs)


    def single_forward(self, img, i, j, h, w, interpolation=InterpolationMode.BILINEAR):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        # i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F2.resized_crop(img, i, j, h, w, self.size, interpolation)

    def forward(self, img, target=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if target is None:
            img = self.single_forward(img, i, j, h, w)
            return img
        elif torch.is_tensor(target):
            img = self.single_forward(img, i, j, h, w)
            target = self.single_forward(target, i, j, h, w, interpolation=InterpolationMode.NEAREST)
            return img, target
        elif isinstance(target,list):
            img = self.single_forward(img, i, j, h, w)
            return_list=[]
            for item in target:
                return_list.append(self.single_forward(item, i, j, h, w, interpolation=InterpolationMode.NEAREST))
            return img, return_list
                 

class ToTensor(T.ToTensor):
    def __init__(self):
        super().__init__()

    def __call__(self, img, target=None):
        if target is None:
            img = F2.to_tensor(img)
            return img
        elif torch.is_tensor(target):
            img = F2.to_tensor(img)
            target = F2.to_tensor(target)
            return img, target
        elif isinstance(target,list):
            img = F2.to_tensor(img)
            return_list=[]
            for item in target:
                return_list.append(F2.to_tensor(item))
            return img,return_list


class Compose(T.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, target=None):
        if target is None:
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                img, target = t(img, target)
            return img, target
