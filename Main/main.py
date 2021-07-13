# main.py

# Load dataset
# Vizualize data
# Transformations
# Defining training loop

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# Loss optimizer function
def loss_optim(model, lr=0.001, momentum=0.9,weight_decay=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    return criterion, optimizer

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

def display_sample(trainloader, count = 25):
    fig = plt.figure(figsize=(15, 5))
    for imgs, labels in trainloader:
        for i in range(count):
            ax = fig.add_subplot(int(count/8)+1, 8, i + 1, xticks=[], yticks=[])
            ax.set_title(f'{classes[labels[i]]}')
            image = imgs[i].cpu()/2 + 0.5
            image = image.numpy().transpose(1, 2, 0).astype(np.float)
            plt.imshow(image)
            plt.tight_layout()
        break

# Data Augmentation
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from torchvision import transforms

def albumentations_transforms(p=1.0, is_train=False):
	# Mean and standard deviation of train dataset
	mean = np.array([0.49139968, 0.48215841, 0.44653091])
	std = np.array([0.24703223, 0.24348513, 0.26158784])
	transforms_list = []
	# Use data aug only for train data
	if is_train:
		transforms_list.extend([
			A.HorizontalFlip(),
            A.ShiftScaleRotate(rotate_limit=15),
            A.Rotate(limit = 5),
            A.CropAndPad(percent= -0.2, keep_size=True, p = 0.25),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=[0.49139968, 0.48215841, 0.44653091], p=0.5),
            A.ToGray(p=0.25)
		])
	transforms_list.extend([A.Normalize(mean=mean,std=std),
                            ToTensorV2()
                         ])
	transformed = A.Compose(transforms_list)
	return lambda img:transformed(image=np.array(img))["image"]
	# return A.Compose(transforms_list)
