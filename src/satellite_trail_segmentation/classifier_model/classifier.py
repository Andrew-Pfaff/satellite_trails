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