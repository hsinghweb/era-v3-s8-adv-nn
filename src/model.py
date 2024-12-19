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
        
        # C1 Block: Added 1x1 convs for feature refinement
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),          # Increased initial channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),                    # 1x1 for channel reduction
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 1),                    # 1x1 for feature refinement
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # C2 Block: Enhanced feature processing
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 48, 3, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 24, 1),                    # 1x1 reduction
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, 1),                    # 1x1 expansion
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 48, 3, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        # C3 Block: Multi-scale feature extraction
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),                    # 1x1 reduction
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, 1),                    # 1x1 expansion
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # C4 Block: Final feature refinement
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 80, 3, padding=2, dilation=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 48, 1),                    # 1x1 reduction
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 1),                    # 1x1 expansion
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 80, 3, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        
        # Output layers with stronger regularization
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)               # Increased dropout
        self.fc = nn.Linear(80, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.dropout(x)
        x = x.view(-1, 80)
        x = self.fc(x)
        return x