import torch
import torch.nn as nn
import torch.nn.init as init


class UNet(nn.Module):
    """
    U-Net CNN architecture for detecting satellite trails.

    The model uses zero padding instead of valid padding so the output mask matches the spatial dimensions of the input image.
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, base_channels=8, dropout=0.0):
        """
        Initializes a configurable U-Net segmentation model.

        Parameters:
            in_channels (int): Number of channels in the input image tensor
            out_channels (int): Number of channels in the output mask tensor
            kernel_size (int): Convolution kernel size used in each block
            base_channels (int): Number of output channels in the first encoder block
            dropout (float): Spatial dropout probability applied after each activation
        """
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.base_channels = base_channels
        self.dropout = dropout


        # Encoder
        self.down1 = self._conv_block(self.in_channels, self.base_channels)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.down2 = self._conv_block(self.base_channels, self.base_channels * 2)
        self.down3 = self._conv_block(self.base_channels * 2, self.base_channels * 4)
        self.down4 = self._conv_block(self.base_channels * 4, self.base_channels * 8)
        self.down5 = self._conv_block(self.base_channels * 8, self.base_channels * 16)

        # Decoder
        self.up_conv1 = nn.ConvTranspose2d(self.base_channels * 16, self.base_channels * 8, kernel_size=2, stride=2)
        self.up1 = self._conv_block(self.base_channels * 16, self.base_channels * 8)
        self.up_conv2 = nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, kernel_size=2, stride=2)
        self.up2 = self._conv_block(self.base_channels * 8, self.base_channels * 4)
        self.up_conv3 = nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, kernel_size=2, stride=2)
        self.up3 = self._conv_block(self.base_channels * 4, self.base_channels * 2)
        self.up_conv4 = nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, kernel_size=2, stride=2)
        self.up4 = self._conv_block(self.base_channels*2, self.base_channels)
        self.final = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self._initialize_weights() #He/Kaiming initialization


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use kaiming_normal_ for weights
                init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for BatchNorm
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


    def _dropout_layer(self):
        """
        Builds the dropout layer used inside each convolution block.

        Returns:
            layer (nn.Module): Dropout layer when dropout > 0, otherwise an identity layer
        """
        if self.dropout == 0:
            return nn.Identity()

        return nn.Dropout2d(p=self.dropout)


    def _conv_block(self, conv_input_channels, conv_output_channels):
        """
        Builds a two-layer convolution block for the encoder or decoder.

        Parameters:
            conv_input_channels (int): Number of channels entering the block
            conv_output_channels (int): Number of channels produced by the block

        Returns:
            block (nn.Sequential): Two-convolution block with normalization, activation, and optional dropout
        """
        return nn.Sequential(
            nn.Conv2d(conv_input_channels, conv_output_channels, self.kernel_size, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(conv_output_channels),
            nn.LeakyReLU(inplace=True),
            self._dropout_layer(),
            nn.Conv2d(conv_output_channels, conv_output_channels, self.kernel_size, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(conv_output_channels),
            nn.LeakyReLU(inplace=True),
            self._dropout_layer(),
        )


    def forward(self,image):
        """
        Passes a batch of images through the U-Net.

        Parameters:
            image (torch.Tensor): Tensor with shape (batch_size, in_channels, height, width)

        Returns:
            output (torch.Tensor): Tensor with shape (batch_size, out_channels, height, width)
        """
        if image.ndim != 4:
            raise ValueError(f"Expected a 4D input tensor, got shape {tuple(image.shape)}")
        if (image.shape[2] % 16) != 0 or (image.shape[3] % 16) != 0:
            raise ValueError(f"Input spatial dimensions {(image.shape[2], image.shape[3])} must be divisible by 16")

        # Pass data throught forward pass
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
        cat1 = torch.cat([uc1, d4], dim=1)
        u1 = self.up1(cat1)
        uc2 = self.up_conv2(u1)
        cat2 = torch.cat([uc2, d3], dim=1)
        u2 = self.up2(cat2)
        uc3 = self.up_conv3(u2)
        cat3 = torch.cat([uc3, d2], dim=1)
        u3 = self.up3(cat3)
        uc4 = self.up_conv4(u3)
        cat4 = torch.cat([uc4, d1], dim=1)
        u4 = self.up4(cat4)
        final = self.final(u4)

        return final
