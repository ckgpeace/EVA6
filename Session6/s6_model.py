import torch.nn.functional as F
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, norm = 'BN', dropout_value = 0.1):
        super(Net, self).__init__()
        
        # INPUT BLOCK
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        if norm == 'BN':
            # CONVOLUTION BLOCK 1
            self.convblock2 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Dropout(dropout_value)
            ) # output_size = 28
            self.convblock3 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Dropout(dropout_value)
            ) # output_size = 26
            self.convblock4 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(12),
                nn.Dropout(dropout_value)
            ) # output_size = 24
        

            # TRANSITION BLOCK 1
            self.convblock5 = nn.Sequential(
                nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            ) # output_size = 24
            self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12


            # CONVOLUTION BLOCK 2
            self.convblock6 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.BatchNorm2d(8),
                nn.Dropout(dropout_value)
            ) # output_size = 10
            self.convblock7 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.BatchNorm2d(12),
                nn.Dropout(dropout_value)
            ) # output_size = 8
            self.convblock8 = nn.Sequential(
                nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Dropout(dropout_value)
            ) # output_size = 6      
            self.convblock9 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.BatchNorm2d(16),
                nn.Dropout(dropout_value)
            ) # output_size = 4
        
        elif norm == 'GN':
            # CONVOLUTION BLOCK 1
            self.convblock2 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(),
                nn.GroupNorm(1, 8),
                nn.Dropout(dropout_value)
            ) # output_size = 28
            self.convblock3 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.GroupNorm(1, 8),
                nn.Dropout(dropout_value)
            ) # output_size = 26
            self.convblock4 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.GroupNorm(1, 12),
                nn.Dropout(dropout_value)
            ) # output_size = 24

            # TRANSITION BLOCK 1
            self.convblock5 = nn.Sequential(
                nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            ) # output_size = 24
            self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12


            # CONVOLUTION BLOCK 2
            self.convblock6 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.GroupNorm(1, 8),
                nn.Dropout(dropout_value)
            ) # output_size = 10
            self.convblock7 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.GroupNorm(1, 12),
                nn.Dropout(dropout_value)
            ) # output_size = 8
            self.convblock8 = nn.Sequential(
                nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.GroupNorm(1, 16),
                nn.Dropout(dropout_value)
            ) # output_size = 6      
            self.convblock9 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.GroupNorm(1, 16),
                nn.Dropout(dropout_value)
            ) # output_size = 4
        
        elif norm == 'LN':
            # CONVOLUTION BLOCK 1
            self.convblock2 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(),
                nn.LayerNorm([8,28,28], elementwise_affine=False),
                nn.Dropout(dropout_value)
            ) # output_size = 28
            self.convblock3 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.LayerNorm([8,26,26], elementwise_affine=False),
                nn.Dropout(dropout_value)
            ) # output_size = 26
            self.convblock4 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.LayerNorm([12,24,24], elementwise_affine=False),
                nn.Dropout(dropout_value)
            ) # output_size = 24

            # TRANSITION BLOCK 1
            self.convblock5 = nn.Sequential(
                nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            ) # output_size = 24
            self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12


            # CONVOLUTION BLOCK 2
            self.convblock6 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.LayerNorm([8,10,10], elementwise_affine=False),
                nn.Dropout(dropout_value)
            ) # output_size = 10
            self.convblock7 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.LayerNorm([12,8,8], elementwise_affine=False),
                nn.Dropout(dropout_value)
            ) # output_size = 8
            self.convblock8 = nn.Sequential(
                nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),
                nn.LayerNorm([16,6,6], elementwise_affine=False),
                nn.Dropout(dropout_value)
            ) # output_size = 6      
            self.convblock9 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                nn.ReLU(),            
                nn.LayerNorm([16,4,4], elementwise_affine=False),
                nn.Dropout(dropout_value)
            ) # output_size = 4
        else:
            raise ValueError("Select correct Norm")

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)

        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)

        x = self.convblock5(x)
        x = self.pool1(x)

        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)

        x = self.gap(x)        
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)