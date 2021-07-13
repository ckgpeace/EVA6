# Train RESNET on Cifar-10 dataset

#### CIFAR-10 has 10 clasess
#### It has 60,000 training images
#### Image size is 3x32x32

## Examples from the CIFAR-10 data

![image](https://user-images.githubusercontent.com/16939617/125388484-2ee4c200-e3bd-11eb-9ae8-e00cb55ccd6d.png)


## About the notebook

1. This notebook imports from 3 custom files - `main.py`, `resnet.py`, `utils.py`
2. `main.py` file contains 
    
    Loss optimizer function - `loss_optim`
    
    Display sample images - `display_sample`
    
    Data Augmentation - `albumentations_transforms`
    
3. `utils.py` file contains
    
   Trainig function - `train`
   
   Test function - `test`
   
   Classwise accuracy - `classwise_acc`
   
   Ploting misclassified images - `plot_misclassified_images`
   
   Validationd and Accuracy Plot - `plot_graph`
   
4. `resnet.py` file contains resnet architecture for resnet18 and resnet34
    

## Model summary of Resnet18

![image](https://user-images.githubusercontent.com/16939617/125373449-c983d780-e3a2-11eb-96a1-a239c427743f.png)

## Misclassified Images
![image](https://user-images.githubusercontent.com/16939617/125388490-3310df80-e3bd-11eb-8f9b-9f603eb698b0.png)

## Test Loss
![image](https://user-images.githubusercontent.com/16939617/125388523-4328bf00-e3bd-11eb-92bd-65b105ef3b41.png)

## Test Accuracy
![image](https://user-images.githubusercontent.com/16939617/125388558-520f7180-e3bd-11eb-9013-ad1b8e3e1f08.png)



