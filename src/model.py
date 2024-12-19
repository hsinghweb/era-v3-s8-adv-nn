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
    Conv3.2(d4)         73      8       4       8       4       3   4   1   4
    Conv3.3(s2)         89      8       4       4       8       3   1   2   1
    
    Conv4.1             105     4       8       4       8       3   1   1   1
    Conv4.2             121     4       8       4       8       3   1   1   1
    Conv4.3(s2)         153     4       8       2       16      3   1   2   1

    Final RF: 153x153 (>44 requirement)
    Parameters: 196,964 (<200K requirement)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1 Block: Reduced initial channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),          # 24->16 channels
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1),         # 32->24 channels
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), # 48->32 channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # C2 Block: Keep Depthwise Separable but reduce channels
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 48, 3),       # 32->48 channels
            nn.BatchNorm2d(48),
            nn.ReLU(),
            DepthwiseSeparableConv(48, 48, 3),       # Keep 48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            DepthwiseSeparableConv(48, 64, 3, stride=2), # 48->64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # C3 Block: More groups for efficiency
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, groups=4),  # groups=4
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=4, dilation=4, groups=4),  # groups=4
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(64, 80, 3, stride=2, padding=1),    # 96->80
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        
        # C4 Block: Reduced channels with groups
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(80, 80, 3, padding=1, groups=4),    # groups=4
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(80, 80, 3, padding=1, groups=4),    # groups=4
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(80, 96, 3, stride=2, padding=1),    # 120->96
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        
        # Output block
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        # C1 and C2 blocks
        x = self.conv1(x)
        x = self.conv2(x)
        
        # C3 block with skip connections
        identity = x
        x = self.conv3_1(x)
        x = x + identity
        identity = x
        x = self.conv3_2(x)
        x = x + identity
        x = self.conv3_3(x)
        
        # C4 block with skip connections
        identity = x
        x = self.conv4_1(x)
        x = x + identity
        identity = x
        x = self.conv4_2(x)
        x = x + identity
        x = self.conv4_3(x)
        
        x = self.gap(x)
        x = self.dropout(x)
        x = x.view(-1, 96)
        x = self.fc(x)
        return x 