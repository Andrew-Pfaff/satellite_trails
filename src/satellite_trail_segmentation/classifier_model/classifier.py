import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet18


def get_classifier_model():
    model = resnet18(weights=None)

    # Adapt for Grayscale (1 channel)
    # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

    # 2. Adapt for Binary Classification
    model.fc = nn.Linear(model.fc.in_features, 1)

    # 3. Apply your He Initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
            
    return model


import torch
import torch.nn as nn


class TrailClassifier(nn.Module):
    """
    Lightweight CNN binary classifier for detecting whether a patch contains a satellite trail.
    """

    def __init__(self, in_channels=1, kernel_size=3, base_channels=16, dropout=0.3):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.base_channels = base_channels
        self.dropout = dropout

        channels = [
            self.base_channels,
            self.base_channels * 2,
            self.base_channels * 4,
            self.base_channels * 8,
        ]

        self.features = nn.Sequential(
            self._conv_block(self.in_channels, channels[0]),
            self._conv_block(channels[0], channels[1]),
            self._conv_block(channels[1], channels[2]),
            self._conv_block(channels[2], channels[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(32, 1),
        )

    def _conv_block(self, conv_input_channels, conv_output_channels):
        return nn.Sequential(
            nn.Conv2d(conv_input_channels, conv_output_channels, kernel_size=self.kernel_size, padding=1, stride=1),
            nn.BatchNorm2d(conv_output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, image):
        if image.ndim != 4:
            raise ValueError(f"Expected a 4D input tensor, got shape {tuple(image.shape)}")
        if (image.shape[2] % 16) != 0 or (image.shape[3] % 16) != 0:
            raise ValueError(f"Input spatial dimensions {(image.shape[2], image.shape[3])} must be divisible by 16")

        features = self.features(image)
        pooled = self.pool(features)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.head(flattened)