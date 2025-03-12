#here we cover transforming data to make it suitable for training 
#All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify the labels
#they accept callables containing the transformation logic
#torchvision.transforms module offers several commonly-used transforms out of the box.

#For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors

#ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. and scales the image’s pixel intensity values in the range [0., 1.]
#Lambda transforms apply any user-defined lambda function.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
#here Lambda define a function to turn the integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.

