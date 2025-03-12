#here we create a custom data set from files

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image

#a custom Dataset class must implement 3 funtions: __init__, __len__, and __getitem__

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) #labels for images stored in csv 
        self.img_dir = img_dir #directory where images are stored 
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): 
        return len(self.img_labels) #returns number of samples in dataset

    def __getitem__(self, idx): #loads and returns a sample from the dataset at the given index idx
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
