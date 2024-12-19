import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from model import CIFAR10Net
from dataset import CIFAR10Dataset
from config import Config
from utils import train_epoch, validate

def display_model_info(model):
    """Display model's RF and parameter count"""
    print("\nReceptive Field Calculation:")
    print("Layer               RF      n_in    j_in    n_out   j_out   k   d   s   p")
    print("Input               1       32      1       32      1       -   -   -   -")
    print("Conv1.1             3       32      1       32      1       3   1   1   1")
    print("Conv1.2             5       32      1       32      1       3   1   1   1")
    print("")
    print("DWConv2(depth)      7       32      1       32      1       3   1   1   1")
    print("DWConv2(point)      7       32      1       32      1       1   1   1   0")
    print("")
    print("Conv3.1(dilated)    15      32      1       32      1       3   4   1   4")
    print("Conv3.2(dilated)    23      32      1       32      1       3   4   1   4")
    print("")
    print("Conv4.1(stride=2)   31      32      1       16      2       3   1   2   1")
    print("Conv4.2(stride=2)   47      16      2       8       4       3   1   2   1")
    print("\nFinal RF: 47x47")
    
    # Print model summary
    print("\nModel Parameter Count:")
    summary(model, (3, 32, 32))

def main():
    # Display CUDA information
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Using Device: {Config.DEVICE}\n")

    # Create datasets and dataloaders
    train_dataset = CIFAR10Dataset(root=Config.DATA_ROOT, train=True)
    test_dataset = CIFAR10Dataset(root=Config.DATA_ROOT, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = CIFAR10Net(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    # Display model information
    display_model_info(model)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = validate(model, test_loader, criterion)
        
        print(f"Epoch: {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    main() 