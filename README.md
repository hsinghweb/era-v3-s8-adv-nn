# ERA-V3 Session 8: Advanced Neural Network Architectures

This repository implements a custom CNN architecture for CIFAR10 classification with specific architectural requirements.

## Network Architecture

The network follows a C1C2C3C4 architecture where each block contains exactly three convolutions with the last one having stride=2 for downsampling:

