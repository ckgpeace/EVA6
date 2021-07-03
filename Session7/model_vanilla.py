## THIS WILL GO IN MODEL.PY FILE

import torch.nn.functional as F
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, dropout_value = 0.05):
        super(Net, self).__init__()
        
        # INPUT BLOCK
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 32, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 32, RF = 5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32, RF =7
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 32, RF = 9
    

        # TRANSITION BLOCK 1
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
        ) # output_size = 32, RF = 9
        self.pool1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, stride = 2, bias=False) 
        # output_size = 16, RF = 11


        # CONVOLUTION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 16, RF = 15
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16, RF = 19
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 16, RF = 23


        # TRANSITION BLOCK 2
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
        ) # output_size = 16, RF = 23
        self.pool2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride = 2, bias=False) 
        # output_size = 8, RF = 27


        # CONVOLUTION BLOCK 3
        ## Depthwise Separable Convolution
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 8, RF = 35
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 8, RF = 43
        ## Dilated Convolution
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 8, RF = 59


        # OUTPUT BLOCK

        self.convblock13 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        # C1
        x = self.convblock1(x)

        # C2
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)

        # T1
        x = self.convblock5(x)
        x = self.pool1(x)

        # C3
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)

        # T2
        x = self.convblock9(x)
        x = self.pool2(x)

        # C4
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.convblock12(x)

        # Out
        x = self.convblock13(x)
        x = self.gap(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)