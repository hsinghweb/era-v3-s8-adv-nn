import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    """
    Receptive Field calculation:
    Layer   RF      n_in    j_in    n_out   j_out   k   d   s   p
    Input   1       32      1       32      1       -   -   -   -
    Conv1.1 3       32      1       32      1       3   1   1   1
    Conv1.2 5       32      1       32      1       3   1   1   1
    Conv2.1 7       32      1       32      1       3   1   1   1
    Conv2.2 9       32      1       32      1       3   1   1   1
    Conv3.1 13      32      1       32      1       3   2   1   2
    Conv3.2 15      32      1       32      1       3   1   1   1
    Conv4.1 23      32      1       16      2       3   1   2   1
    Conv4.2 47      16      2       16      2       3   1   1   1
    
    Final RF: 47x47
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1 Block - Increased channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # C2 Block with Depthwise Separable Conv
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(64, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            DepthwiseSeparableConv(96, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # C3 Block with Dilated Conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 160, 3, padding=2, dilation=2),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            DepthwiseSeparableConv(160, 192, 3),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        
        # C4 Block with stride=2
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(192, 224, 3, stride=2),
            nn.BatchNorm2d(224),
            nn.ReLU(),
            DepthwiseSeparableConv(224, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Global Average Pooling and Final FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x 