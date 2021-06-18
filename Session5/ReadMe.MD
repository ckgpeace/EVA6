## The result of the Session5 assignemnt: 

### 1. Session5_1.ipynb

#### Target
Reduce the number of parameters (less than 10K) by reducing the channel size in initial blocks as the features extracted are simpler
Remove padding in the last layers of the convulution, padding in last layers where the number of pixels are low can affect the feature extraction as we are augmenting large portion of input.
Need to keep the accuracy near the target accuracy

#### Results
The number of parameters has reduced below 10K - 9,752
The test accuracy is ~99.3 consistently in last 5 epochs out of 15 epochs

#### Analysis
The parameters has reduced but the accuracy has went down
The accuracy decreased as we reduced the capacity of the model


### 2. Session5_2.ipynb

#### Target
Increase the capacity of the 1st Convolution block as it has only one layer
Need to keep the accuracy near the target accuracy

#### Results
The number of parameters has reduced below 10K - 9752
The test accuracy is ~99.15 consistently in last 5 epochs out of 15 epochs

#### Analysis
Although the number of parameters has not changed from previous step, the level of accuracy has decreased in given 15 epochs
It might be possible that model needs longer training time or epochs to achieve higher accuracy

### 3. Session5_3.ipynb

#### Target
Try color jitter transformation to increase accuracy
Reducing the number of parameters from 9,752 to ~8K by reducing channel size of 16 to 12 in some convolution block

#### Results
The number of parameters has reduced below 8,128
The test accuracy is above 99.3 consistently in last 5 epochs out of 15 epochs

#### Analysis
Adding color jitter augmentaion has pulled the accuracy to 99.34 while there is decrease in the number of parameters

### 4. Session5_4.ipynb

#### - Late assignment requirement: rotation(+6.96,-6.95), Parameter < 8K, dropout = 0.05

#### Target
Changing the architecture to accomodate one 16x16 channel to capture complexity while keeping the number of paramaters less than 8K


#### Results
The number of parameters has reduced to 7,996
The test accuracy is above 99.3+ in 15 epoch and consistently around 99.3 in last 4-5 epochs out of 15 epochs

#### Analysis
Using 16 x 16 channel has incresed the accuracy as it can capture more complex feartures from the data
We need small number of channels in initial layers as we intend to capture simpler features like edges and gradients and large number of channels in later later layers to capture complex features like patterns and object


