# Install this for latest 'albumentations'
# !pip install -U git+https://github.com/albu/albumentations --no-cache-dir

# Albumentations Transforms
# Transformations - HorizontalFlip, ShiftScaleRotate, CoarseDropout, ToGray

import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2

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
            A.CoarseDropout(max_holes=1, max_height=16, max_width=1, min_holes=1, min_height=4, min_width=1, fill_value=[0.49139968, 0.48215841, 0.44653091], p=0.25),
            A.ToGray(p=0.25)
		])
	transforms_list.extend([A.Normalize(mean=mean,std=std), 
                            ToTensorV2()
                         ])
	transformed = A.Compose(transforms_list, p=p)
	return lambda img:transformed(image=np.array(img))["image"]