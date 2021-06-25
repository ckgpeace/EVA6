## Batch Normalization
Batch Normalization has revolutionized the CNN through faster and efficient training. BN has tackled the problem of vanishing and exploding gradient problem, this has helped in 
training very deep CNN models


## Layer normalization
Layer normalization (LayerNorm) is a technique to normalize the distributions of intermediate layers. It enables smoother gradients, faster training, and better generalization accuracy
https://arxiv.org/pdf/1911.07013.pdf
Pytorch function: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

## Group Normalization
GN divides the channels into groups and computes within each group the mean and variance for normalization. GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes.
https://arxiv.org/abs/1803.08494


### Normalization methods
![image](https://user-images.githubusercontent.com/16939617/123479317-606c3800-d61e-11eb-8cda-334254bd5f2c.png)

### BN, LN and GN illustraion on 4 image sample 
Excel file is included in the folder

![image](https://user-images.githubusercontent.com/16939617/123479432-8e517c80-d61e-11eb-9684-c4351db6cdb5.png)

### Summary of Session6_1.ipynb 
All 3 normalization techniques are applied with L1 and L2 reguralization
1. Network with Group Normalization + L1
2. Network with Layer Normalization + L2
3. Network with L1 + L2 + BN

Misclassified images and test/train accuracy and losses are also included

### s6_model.py file 
1. It contains the model with all 3 normalizations - LN, BN, GN are implemnted
2. L1 and L2 loss fucntions are also implemented

### Final results
Test accuracy of GN+L1 regularization seems to supersede BN+L1+L2 and LN+L2. GN+L1 has consistent accuracy on last epochs
![image](https://user-images.githubusercontent.com/16939617/123480473-08363580-d620-11eb-9372-153f70d3d89b.png)

### Misclassified images 

1. BN+L1+L2 

![image](https://user-images.githubusercontent.com/16939617/123480551-269c3100-d620-11eb-8203-50f9006d4c52.png)

2. GN+L1

![image](https://user-images.githubusercontent.com/16939617/123480593-33b92000-d620-11eb-8eeb-218d5fa3a036.png)

3. LN+L2

![image](https://user-images.githubusercontent.com/16939617/123480640-45022c80-d620-11eb-9a9b-26c130f0c831.png)






