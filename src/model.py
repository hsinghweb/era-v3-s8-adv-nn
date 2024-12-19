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
    Receptive Field calculation for current architecture:
    Layer               RF      n_in    j_in    n_out   j_out   k   d   s   p
    Input               1       32      1       32      1       -   -   -   -
    Conv1.1             3       32      1       32      1       3   1   1   1
    Conv1.2             5       32      1       32      1       3   1   1   1
    
    DWConv2(depth)      7       32      1       32      1       3   1   1   1
    DWConv2(point)      7       32      1       32      1       1   1   1   0
    
    Conv3.1(dilated)    15      32      1       32      1       3   4   1   4
    Conv3.2(dilated)    23      32      1       32      1       3   4   1   4
    
    Conv4.1(stride=2)   31      32      1       16      2       3   1   2   1
    Conv4.2(stride=2)   47      16      2       8       4       3   1   2   1
    
    Final RF: 47x47
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1 Block - Reduced channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # C2 Block with Depthwise Separable Conv only
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 48, 3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        # C3 Block with two Dilated Conv (dilation=4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, padding=4, dilation=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        
        # C4 Block with two stride=2 convs
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Global Average Pooling and Final FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x 