import torch
import torch.nn as nn
import torch.nn.init as init


class AttentionGate(nn.Module):
    """
    Additive attention gate for filtering U-Net skip connections.

    The skip tensor and decoder gating tensor are projected to a shared
    intermediate channel space, combined additively, converted to a
    single-channel attention map, and multiplied into the skip tensor.
    """

    def __init__(self, skip_channels, gating_channels, intermediate_channels=None):
        super().__init__()

        if intermediate_channels is None:
            intermediate_channels = max(skip_channels // 2, 1)

        self.skip_channels = skip_channels
        self.gating_channels = gating_channels
        self.intermediate_channels = intermediate_channels

        self.skip_projection = nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.gating_projection = nn.Conv2d(gating_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def attention_coefficients(self, skip, gating):
        if skip.shape[2:] != gating.shape[2:]:
            raise ValueError(
                "Skip and gating tensors must have matching spatial dimensions, "
                f"got {tuple(skip.shape[2:])} and {tuple(gating.shape[2:])}"
            )

        return self.attention(self.skip_projection(skip) + self.gating_projection(gating))

    def forward(self, skip, gating):
        coefficients = self.attention_coefficients(skip, gating)
        return skip * coefficients


class AttentionUNet(nn.Module):
    """
    Attention U-Net segmentation model for detecting satellite trails.

    This mirrors the baseline U-Net encoder/decoder and inserts additive
    attention gates on the skip connections. The model returns raw logits;
    apply sigmoid outside the model for probability maps.
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, base_channels=8, dropout=0.0, use_batchnorm=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.base_channels = base_channels
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.model_name = "attention_unet"

        # Encoder
        self.down1 = self._conv_block(self.in_channels, self.base_channels)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.down2 = self._conv_block(self.base_channels, self.base_channels * 2)
        self.down3 = self._conv_block(self.base_channels * 2, self.base_channels * 4)
        self.down4 = self._conv_block(self.base_channels * 4, self.base_channels * 8)
        self.down5 = self._conv_block(self.base_channels * 8, self.base_channels * 16)

        # Decoder
        self.up_conv1 = nn.ConvTranspose2d(self.base_channels * 16, self.base_channels * 8, kernel_size=2, stride=2)
        self.gate1 = AttentionGate(self.base_channels * 8, self.base_channels * 8)
        self.up1 = self._conv_block(self.base_channels * 16, self.base_channels * 8)

        self.up_conv2 = nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, kernel_size=2, stride=2)
        self.gate2 = AttentionGate(self.base_channels * 4, self.base_channels * 4)
        self.up2 = self._conv_block(self.base_channels * 8, self.base_channels * 4)

        self.up_conv3 = nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, kernel_size=2, stride=2)
        self.gate3 = AttentionGate(self.base_channels * 2, self.base_channels * 2)
        self.up3 = self._conv_block(self.base_channels * 4, self.base_channels * 2)

        self.up_conv4 = nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, kernel_size=2, stride=2)
        self.gate4 = AttentionGate(self.base_channels, self.base_channels)
        self.up4 = self._conv_block(self.base_channels * 2, self.base_channels)

        self.final = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self._initialize_weights()

    def _initialize_weights(self):
        """Applies Kaiming initialization to convolution layers."""

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(module.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    def _dropout_layer(self):
        if self.dropout == 0:
            return nn.Identity()

        return nn.Dropout2d(p=self.dropout)

    def _conv_block(self, conv_input_channels, conv_output_channels):
        layers = [
            nn.Conv2d(conv_input_channels, conv_output_channels, self.kernel_size, stride=1, padding=1, dilation=1),
        ]
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(conv_output_channels))
        layers.extend(
            [
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                self._dropout_layer(),
                nn.Conv2d(conv_output_channels, conv_output_channels, self.kernel_size, stride=1, padding=1, dilation=1),
            ]
        )
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(conv_output_channels))
        layers.extend(
            [
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                self._dropout_layer(),
            ]
        )

        return nn.Sequential(*layers)

    def forward(self, image):
        if image.ndim != 4:
            raise ValueError(f"Expected a 4D input tensor, got shape {tuple(image.shape)}")
        if (image.shape[2] % 16) != 0 or (image.shape[3] % 16) != 0:
            raise ValueError(f"Input spatial dimensions {(image.shape[2], image.shape[3])} must be divisible by 16")

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
        d4_att = self.gate1(skip=d4, gating=uc1)
        u1 = self.up1(torch.cat([uc1, d4_att], dim=1))

        uc2 = self.up_conv2(u1)
        d3_att = self.gate2(skip=d3, gating=uc2)
        u2 = self.up2(torch.cat([uc2, d3_att], dim=1))

        uc3 = self.up_conv3(u2)
        d2_att = self.gate3(skip=d2, gating=uc3)
        u3 = self.up3(torch.cat([uc3, d2_att], dim=1))

        uc4 = self.up_conv4(u3)
        d1_att = self.gate4(skip=d1, gating=uc4)
        u4 = self.up4(torch.cat([uc4, d1_att], dim=1))

        return self.final(u4)
