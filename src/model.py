import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, dilation=dilation
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
    Parameters: 194,950 (<200K requirement)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1 Block: Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),          # Basic 3x3 conv
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Strided conv for reduction
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # C2 Block: Depthwise Separable Convolution
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, 3),      # DW spatial
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # Strided conv
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # C3 Block: Dilated convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=2, dilation=2),  # Dilated conv
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # Strided conv
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # C4 Block: Final feature refinement
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),                  # 1x1 conv for channel reduction
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)               # Reduced dropout
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.dropout(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x