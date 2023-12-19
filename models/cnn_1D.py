import torch
from torch import nn

class Cnn1D(torch.nn.Module):    
    def __init__(self, kernel_size, pred_diff, in_channels):
        super().__init__()        
        self.pred_diff = pred_diff
        self.act_func = nn.ReLU()
        self.conv1 = nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels, out_channels=in_channels * 2, padding=2) 
        self.bn1 = nn.BatchNorm1d(in_channels * 2)
        self.conv2 = nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels * 2, out_channels=in_channels * 4, padding=2)
        self.bn2 = nn.BatchNorm1d(in_channels * 4)
        self.conv3 = nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels * 4, out_channels=in_channels * 8, padding=2)
        self.bn3 = nn.BatchNorm1d(in_channels * 8)
        self.conv4 = nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels * 8, out_channels=in_channels * 4, padding=2)
        self.bn4 = nn.BatchNorm1d(in_channels * 4)
        self.conv5 = nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels * 4, out_channels=in_channels * 2, padding=2)
        self.bn5 = nn.BatchNorm1d(in_channels * 2)
        self.conv6 = nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels * 2, out_channels=in_channels, padding=2)
        
    def forward(self, x):
        init = x  # save input for later use (in case we want to predict the diff)

        x = self.bn1(self.act_func(self.conv1(x)))
        x = self.bn2(self.act_func(self.conv2(x)))
        x = self.bn3(self.act_func(self.conv3(x)))
        x = self.bn4(self.act_func(self.conv4(x)))
        x = self.bn5(self.act_func(self.conv5(x)))
        x = self.conv6(x)

        if self.pred_diff:
            x = x + init

        # force output to be in [0,1] (because the transforms assume that).
        x = torch.sigmoid(x) # maybe no act funct at end ... but some transforms require output to be in [0,1] and I am not sure clipping is good for gradients

        return x