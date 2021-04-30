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
import math

import pdb


class DNA_Aug(data.Dataset):
    """DNA_Aug

    Data is from the Laboratory of Systems Pharmocology at 
    Harvard Medical School
    """
    colors = [
        [0, 0, 128],
        [0, 255, 0],
        [0, 128, 255],
    ]

    label_colors = dict(zip(range(3), colors))

    def __init__(self, root, split="train", img_size=(128, 128)):
        """__init__
        :param root:
        :param split:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.n_classes = 3
        self.img_size = img_size
        self.files = {}

        self.images_base = os.path.join(self.root, self.split)

        # Generate list of all tif files
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

        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

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

        wt_path = os.path.join(
            self.images_base,
            os.path.basename(img_path)[:-7] + "wt.tif",
        )        

        # Read in image to numpy array
        img = tifffile.imread(img_path)

        #Intensity (on-the-fly) augmentations to all channels
        maxBrig = 0.22 # stdev of image data set
        maxCont = 0.1 * maxBrig
        fBrig = maxBrig * np.float_power(-1, np.random.rand() < 0.5) * np.random.rand()
        fCont = 1 + maxCont * np.float_power(-1, np.random.rand() < 0.5) * np.random.rand()
        img = img * fCont + fBrig

        # Scaling image into [0, 1] range by dividing by the max
        img = img / float(256-1)
        # stays same for each getitem() call
        rand_seed = np.random.randint(2147483647)
        
        ### DNA with Aug ###
        torch_img = []
        channel = random.choice([0,2,4]) # randomly choose a channel to train from
        for i in range(3):
            ch_img = Image.fromarray(img[channel, ...])
            random.seed(rand_seed)  # apply this seed to img transforms
            torch_img.append(self.image_transform(ch_img))
        torch_img = torch.cat(torch_img, dim=0)

        # Label transform
        np_lbl = tifffile.imread(lbl_path)
        tr_lbl = np.zeros_like(np_lbl, np.uint8)
        for orig_label, train_label in self.valid_classes.items():
            tr_lbl[np_lbl == orig_label] = train_label
        # tr_lbl = cv2.resize(tr_lbl, (800, 800),
        #                     interpolation=cv2.INTER_NEAREST)
        # Apply the same augmentation (using the same rand seed)
        random.seed(rand_seed)

        torch_label = torch.from_numpy(tr_lbl)
        torch_label = torch_label.type(torch.long)

        #Nuclei Weight Transform
        nuclei_weight = np.zeros_like(np_lbl,np.uint8)
        nuclei_weight[np_lbl == 3] = 1

        torch_nuc_wt = torch.from_numpy(nuclei_weight)
        torch_nuc_wt = 2*torch_nuc_wt + 1 # any number to penalize incorrect contour predictions
        torch_nuc_wt = torch_nuc_wt.type(torch.long)

        # Weight Transform
        weight = tifffile.imread(wt_path)
        # weight = cv2.resize(weight, (800, 800),
        #                     interpolation=cv2.INTER_NEAREST)
        torch_wt = torch.from_numpy(weight)
        torch_wt = 7*torch_wt + 1 # any number to penalize incorrect contour predictions
        torch_wt = torch_wt.type(torch.long)

        #torch_label = torch.matmul(torch_label.type(torch.float), torch_wt.type(torch.float)).type(torch.long)
        
        return torch_img, torch_label, torch_wt, torch_nuc_wt

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