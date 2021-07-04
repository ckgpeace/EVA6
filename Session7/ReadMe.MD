# Training CNN from scratch on CIFAR-10 dataset


## Data
### CIFAR-10

#### Train Data has 50,000 images. Test data has 10,000 images

Mean and Std. Dev. of Train imgages for 3 channels:
#### Mean -      [0.4914, 0.4822, 0.4465]
#### Std. Dev -  [0.2470, 0.2435, 0.2616]

*Some Images:
![image](https://user-images.githubusercontent.com/16939617/124369328-58e50880-dc88-11eb-93f2-7e085c7213d2.png)

## Model Summary:

### Total Parameters = 104K

### C10 and C11 are Depthwise separable convolution layers and C13 is dialted convolution
### Padding is '1' for all the layers. Relu-BN-DO after every layer except the last layer
### GAP has been used to get number of classes

Input Block =   C1(in = 3, out = 16, kernel = 3)   -> Relu-BN-DO  #RF = 3 (Receptive Field)  #Size =32 (Output Channel Size)

CNN Block1  =   C2(16,16,3) ->  C3(16,32,3)  ->  C4(32,32,3) #RF = 9 #Size =32

Trans Block1 = C5(32,16,1) -> POOL1(16,16,3, stride = 2) #RF = 11 #Size = 16 

CNN Block2  =   C6(16,16,3) -> C7(16,32,3) -> C8(32,64,3) -> #RF = 23 #Size =16

Trans Block2 = C9(64,32,1) -> POOL1(32,32,3, stride = 2) #RF = 27 #Size = 8 

CNN Block3  =   C10(32,64,3, groups = 32) : followed by 1x1 -> C11(64,64,3, groups = 32) -> C12(64,64,3, dilation = 2)  -> Relu-BN-DO #RF = 23 #Size =16

Output Block = C10(64,10,1) -> GAP(8) -> Softmax

## Data Augmentation 
### 'albumentations' library was used for data augmentaions
Mainly these DA techniques *HorizontalFlip ShiftScaleRotate CoarseDropout ToGray*
There are many more DA techniques in 'albumentations' library but those are for some other day.
https://github.com/albu/albumentations


## Results

### Number of Epochs = 50
### Average Test Accuracy in last 5 EPOCHS = *85%*
### Class Accuracy at 85% test Accruray:

Accuracy of plane : 89 %

Accuracy of   car : 97 %

Accuracy of  bird : 83 %

Accuracy of   cat : 65 %

Accuracy of  deer : 79 %

Accuracy of   dog : 75 %

Accuracy of  frog : 85 %

Accuracy of horse : 89 %

Accuracy of  ship : 92 %

Accuracy of truck : 86 %


### Log for last 5 epochs
![image](https://user-images.githubusercontent.com/16939617/124369453-a0b85f80-dc89-11eb-9817-df8c69b18751.png)

## Plots - Validation Accuracy and Valdidation loss
![image](https://user-images.githubusercontent.com/16939617/124369467-c6ddff80-dc89-11eb-9f29-dedfe761f692.png)

![image](https://user-images.githubusercontent.com/16939617/124369473-ce050d80-dc89-11eb-9932-2f75a4408741.png)


## Mis-classified images
![image](https://user-images.githubusercontent.com/16939617/124369489-e07f4700-dc89-11eb-9456-3e84fc425c4c.png)




