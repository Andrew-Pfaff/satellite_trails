import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net CNN architecture for detecting satellite trails with zero padding to preserve spatial dimensions."""

    def __init__(self, kernel_size, stride=1, padding=1, dilation=1, base_channels=16, depth=3, in_channels=1, out_channels=1, dropout=0.0):
        """
        Initializes a configurable U-Net segmentation model.
        Parameters:
            kernel_size (int): Convolution kernel size used in each block
            stride (int): Convolution stride used inside each block and in the final layer
            padding (int): Padding applied to each convolution
            dilation (int): Dilation applied to each convolution
            base_channels (int): Number of output channels in the first encoder block
            depth (int): Number of downsampling stages before the bottleneck
            in_channels (int): Number of channels in the input image tensor
            out_channels (int): Number of channels in the output mask tensor
            dropout (float): Spatial dropout probability applied after each activation
        """
        super().__init__()

        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if stride != 1:
            raise ValueError(f"stride must be 1 for this U-Net implementation, got {stride}")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.base_channels = base_channels
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        encoder_channels = [base_channels * (2 ** level) for level in range(depth + 1)]

        self.down_blocks = nn.ModuleList()
        current_in_channels = in_channels
        for output_channels in encoder_channels:
            self.down_blocks.append(self._conv_block(current_in_channels, output_channels))
            current_in_channels = output_channels

        self.pool = nn.MaxPool2d(2, stride=2)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for input_channels, skip_channels in zip(reversed(encoder_channels[1:]), reversed(encoder_channels[:-1])):
            self.up_convs.append(nn.ConvTranspose2d(input_channels, skip_channels, kernel_size=2, stride=2))
            self.up_blocks.append(self._conv_block(skip_channels * 2, skip_channels))

        self.final = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1, stride=self.stride, padding=0, dilation=self.dilation)

    def _regularization_layer(self):
        """
        Builds the regularization layer used inside each convolution block.
        Returns:
            layer (nn.Module): Dropout layer when dropout > 0, otherwise an identity layer
        """
        if self.dropout == 0:
            return nn.Identity()

        return nn.Dropout2d(p=self.dropout)

    def _conv_block(self, input_channels, output_channels):
        """
        Builds a two-layer convolution block for the encoder or decoder.
        Parameters:
            input_channels (int): Number of channels entering the block
            output_channels (int): Number of channels produced by the block
        Returns:
            block (nn.Sequential): Two-convolution block with normalization, activation, and optional dropout
        """
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, self.kernel_size, self.stride, self.padding, self.dilation),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            self._regularization_layer(),
            nn.Conv2d(output_channels, output_channels, self.kernel_size, self.stride, self.padding, self.dilation),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            self._regularization_layer(),
        )

    def forward(self, image):
        """
        Passes an image tensor through the U-Net.
        Parameters:
            image (torch.Tensor): Tensor with shape (batch_size, in_channels, height, width)
        Returns:
            output (torch.Tensor): Tensor with shape (batch_size, out_channels, height, width)
        """
        if image.ndim != 4:
            raise ValueError(f"Expected a 4D input tensor, got shape {tuple(image.shape)}")
        if image.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {image.shape[1]}")

        downsample_factor = 2 ** self.depth
        height, width = image.shape[-2:]
        if height % downsample_factor != 0 or width % downsample_factor != 0:
            raise ValueError(f"Input spatial dimensions {(height, width)} must be divisible by 2**depth={downsample_factor}")

        skip_connections = []
        features = image

        for down_block in self.down_blocks[:-1]:
            features = down_block(features)
            skip_connections.append(features)
            features = self.pool(features)

        features = self.down_blocks[-1](features)

        for up_conv, up_block, skip_features in zip(self.up_convs, self.up_blocks, reversed(skip_connections)):
            features = up_conv(features)

            if features.shape[-2:] != skip_features.shape[-2:]:
                raise ValueError(f"Upsampled tensor shape {tuple(features.shape[-2:])} does not match skip tensor shape {tuple(skip_features.shape[-2:])}")

            features = torch.cat([features, skip_features], dim=1)
            features = up_block(features)

        output = self.final(features)

        return output
