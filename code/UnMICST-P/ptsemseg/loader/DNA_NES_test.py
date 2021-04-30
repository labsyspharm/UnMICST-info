import os
import copy
import h5py
import torch
import random
import tifffile

from PIL import Image
from torch.utils import data
from ptsemseg.utils import recursive_glob

import numpy as np
import scipy.misc as m
import torchvision.transforms as T

import matplotlib.pyplot as plt

import cv2

import pdb

class DNA_NES_test(data.Dataset):
    """DNA_NES_test

    Data is from the Laboratory of Systems Pharmocology at 
    Harvard Medical School
    """
    colors = [
        [0, 0, 128],
        [0, 255, 0],
        [0, 128, 255],
    ]

    label_colors = dict(zip(range(3), colors))
    
    def __init__(self, root, split="test"):
        """__init__
        :param root:
        :param split:
        :param setting:
        """        
        
        self.root = root
        self.split = split
        self.n_classes = 3
        self.files = {}
        
        self.images_base = os.path.join(self.root, self.split)
        
        # Generate list of all tif files and save in dictionary
        self.files[split] = recursive_glob(
            rootdir=self.images_base, suffix="_Img.tif")

        self.ignore_index = 250
        self.valid_classes = {
            0: self.ignore_index,
            1: 0,
            2: 1,
            3: 2}
        self.class_names = {
            self.ignore_index: "Ignore",
            0: "Background",
            1: "Contour",
            2: "Nuclei"}

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (
                split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

        ## Augmentations ###
        self.image_transform = self.get_transform(split == "train")

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """

        img_path = self.files[self.split][index].rstrip()

        lbl_path = os.path.join(
        self.images_base,
        os.path.basename(img_path)[:-7] + "Ant.tif",
        )

        img_name = os.path.basename(img_path)[:-8]

        # Read in image to numpy array
        img = tifffile.imread(img_path)

        # Scaling image into [0, 1] range by dividing by the max
        img = img / float(256-1)
        # stays same for each getitem() call
        rand_seed = np.random.randint(2147483647)
        
        # Test DNA+NES models
        torch_images = {}
        for channel in range(7):
            torch_img = []
            for j in range(3):
                if (j == 2): # Last channel is Lamin
                    ch_img = Image.fromarray(img[6, ...])
                elif (j == 1): # Add zero channel
                    ch_img = np.zeros((np.shape(img)[1],np.shape(img)[2]))
                    ch_img = Image.fromarray(ch_img) #convert back to PIL image 
                else:
                    ch_img = Image.fromarray(img[channel, ...])
                random.seed(rand_seed)  # apply this seed to img transforms
                torch_img.append(self.image_transform(ch_img))
            torch_img = torch.cat(torch_img, dim=0)
            
            torch_name = img_name + "_" + str(channel) # Example: 'I00018_2'

            torch_images[torch_name] = torch_img

        # Label transform
        np_lbl = tifffile.imread(lbl_path)
        tr_lbl = np.zeros_like(np_lbl, np.uint8)
        for orig_label, train_label in self.valid_classes.items():
            tr_lbl[np_lbl == orig_label] = train_label
        # Apply the same augmentation (using the same rand seed)
        random.seed(rand_seed)

        torch_label = torch.from_numpy(tr_lbl)
        torch_label = torch_label.type(torch.long)
                      
        return torch_images, torch_label

    def get_transform(self, is_train):
        transforms = [] 
        transforms.append(T.Resize(size=256, interpolation=Image.BICUBIC))
        # from PIL to torch.Tensor
        transforms.append(T.ToTensor())
        # subtract 0.5 because pixel values are in range [0, 1]
        transforms.append(T.Normalize(mean=(0.5,), std=(0.25,)))
        #transforms.append(T.Normalize(mean=(0.28,), std=(0.22,))) # pre-computed dataset mean and SD
        
        # returns a function applied to each channel
        return T.Compose(transforms)

    def decode_segmap(self, temp):
        # Create color map for segmentation mask
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()

        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colors[l][0]
            g[temp == l] = self.label_colors[l][1]
            b[temp == l] = self.label_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb
        
