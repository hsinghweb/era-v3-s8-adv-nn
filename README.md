# ERA-V3 Session 8: Advanced Neural Network Architectures

This repository implements a custom CNN architecture for CIFAR10 classification with specific architectural requirements.

## Network Architecture

The network follows a C1C2C3C4 architecture where each block contains exactly three convolutions with the last one having stride=2 for downsampling:

python
Architecture:
C1 Block: Regular Convolutions
- Conv1.1: 3×3, channels: 3→16
- Conv1.2: 3×3, channels: 16→16
- Conv1.3: 3×3, stride=2, channels: 16→24

C2 Block: Depthwise Separable Convolutions
- DWConv2.1: 3×3, channels: 24→32
- DWConv2.2: 3×3, channels: 32→32
- DWConv2.3: 3×3, stride=2, channels: 32→48

C3 Block: Dilated Convolutions
- Conv3.1: 3×3, dilation=2, channels: 48→48
- Conv3.2: 3×3, dilation=2, channels: 48→48
- Conv3.3: 3×3, stride=2, channels: 48→64

C4 Block: Regular Convolutions
- Conv4.1: 3×3, channels: 64→64
- Conv4.2: 3×3, channels: 64→64
- Conv4.3: 3×3, stride=2, channels: 64→72

Output:
- Global Average Pooling
- FC Layer: 72→10 classes

## Meeting Architecture Requirements

1. **C1C2C3C4 with 3 Convolutions**:
   - Each block has exactly 3 convolutions
   - Last convolution in each block uses stride=2 instead of MaxPooling
   - Gradual channel increase: 3→24→48→64→72

2. **Receptive Field > 44**:
   - Final RF: 137×137
   - Achieved through combination of:
     - Regular convolutions
     - Dilated convolutions
     - Strided convolutions
     - Accumulated jump factors

3. **Depthwise Separable Convolution**:
   - Implemented in C2 block
   - Reduces parameters while maintaining performance
   - All three convolutions in C2 are depthwise separable

4. **Dilated Convolution**:
   - Implemented in C3 block
   - First two convolutions use dilation=2
   - Helps increase RF without increasing parameters

5. **GAP and FC**:
   - Global Average Pooling after C4 block
   - Single FC layer to map to 10 classes
   - Reduces parameters compared to multiple FC layers

6. **Albumentation Transformations**:
   - Horizontal Flip (p=0.5)
   - ShiftScaleRotate
   - CoarseDropout with specified parameters
   - All transformations applied during training only

## Receptive Field Calculation

The RF grows through the network as follows:
```
Layer               RF    Calculation
Input               1     Initial RF
Conv1.1-1.3         9     Regular convs with stride=2 in last
DWConv2.1-2.3       25    Effect of previous stride
Conv3.1-3.3(d2)     73    Dilated convs + stride effect
Conv4.1-4.3         137   Final RF with accumulated jump
```

## Parameter Count (194,950 total)

Distribution of parameters:
1. C1 Block: ~6,360 (3.7%)
   - Regular convolutions
   - Initial feature extraction

2. C2 Block: ~4,544 (2.6%)
   - Depthwise separable convs
   - Parameter efficient

3. C3 Block: ~69,600 (40.5%)
   - Dilated convolutions
   - Larger RF coverage

4. C4 Block: ~89,696 (52.2%)
   - Regular convolutions
   - Final feature refinement

5. FC Layer: ~750 (0.4%)
   - Final classification

## Training Features

1. **Device Handling**:
   - Automatic CUDA detection
   - Device-agnostic training

2. **Data Management**:
   - Custom CIFAR10Dataset class
   - Albumentation transformations
   - Efficient data loading with DataLoader

3. **Training Loop**:
   - Tracks both training and test metrics
   - Displays progress with tqdm
   - Saves best model based on test accuracy

4. **Model Information**:
   - Detailed RF calculation display
   - Architecture verification
   - Parameter count tracking

5. **Optimization**:
   - Adam optimizer
   - Cross Entropy Loss
   - Learning rate: 0.001
   - Batch size: 128

