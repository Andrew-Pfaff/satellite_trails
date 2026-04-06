import os
import logging

import torch
import torch.nn as nn



class UNet(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=1, dilation=1):
        """
        UNet CNN architecture for detecting satelite trails.
        Uses zero padding instead of valid padding for equal dimensionality of output mask.

        Parameters:

        """
        
        super(UNet, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        def conv(input_channels, output_channels):
            return nn.Sequential(nn.Conv2d(input_channels, output_channels, self.kernel_size, self.stride, self.padding, self.dilation),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(output_channels, output_channels, self.kernel_size, self.stride, self.padding, self.dilation),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU())


        #U-net architecture
        self.down1 = conv(1,64)
        self.pool = nn.MaxPool2d(2,stride=2)
        self.down2 = conv(64,128)
        #Pool
        self.down3 = conv(128,256)
        #Pool
        self.down4 = conv(256,512)
        #Pool
        self.down5 = conv(512,1024)
        self.up_conv1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        #Cat
        self.up1 = conv(1024,512)
        self.up_conv2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        #Cat
        self.up2 = conv(512,256)
        self.up_conv3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        #Cat
        self.up3 = conv(256,128)
        self.up_conv4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        #Cat
        self.up4 = conv(128,64)
        self.final = nn.Conv2d(64, 1, kernel_size=1, stride = self.stride, padding=0, dilation=self.dilation)


    def forward(self,image):
        d1 = self.down1(image)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        p4 = self.pool(d4)
        d5 = self.down5(p4)

        uc1 = self.up_conv1(d5)
        cat1 = torch.cat([uc1,d4],dim=1)
        u1 = self.up1(cat1)
        uc2 = self.up_conv2(u1)
        cat2 = torch.cat([uc2,d3],dim=1)
        u2 = self.up2(cat2)
        uc3 = self.up_conv3(u2)
        cat3 = torch.cat([uc3,d2],dim=1)
        u3 = self.up3(cat3)
        uc4 = self.up_conv4(u3)
        cat4 = torch.cat([uc4,d1],dim=1)
        u4 = self.up4(cat4)
        final = self.final(u4)        

        return final