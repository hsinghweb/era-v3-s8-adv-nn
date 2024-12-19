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
    Conv1.3(s2)         9       32      1       16      2       3   1   2   1
    
    DWConv2.1           13      16      2       16      2       3   1   1   1
    DWConv2.2           17      16      2       16      2       3   1   1   1
    DWConv2.3(s2)       25      16      2       8       4       3   1   2   1
    
    Conv3.1(d2)         41      8       4       8       4       3   2   1   2
    Conv3.2(d2)         57      8       4       8       4       3   2   1   2
    Conv3.3(s2)         73      8       4       4       8       3   1   2   1
    
    Conv4.1             89      4       8       4       8       3   1   1   1
    Conv4.2             105     4       8       4       8       3   1   1   1
    Conv4.3(s2)         137     4       8       2       16      3   1   2   1

    Final RF: 137x137 (>44 requirement)
    Parameters: 196,964 (<200K requirement)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1 Block with three conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        
        # C2 Block with Depthwise Separable Conv
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(24, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 48, 3, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        # C3 Block with Dilated Conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=2, dilation=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=2, dilation=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # C4 Block - Reduced channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 72, 3, stride=2, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(),
        )
        
        # Global Average Pooling and Final FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(72, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 72)
        x = self.fc(x)
        return x 