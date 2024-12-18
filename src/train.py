import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from model import CIFAR10Net
from dataset import CIFAR10Dataset
from config import Config
from utils import train_epoch, validate

def main():
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
    
    # Print model summary
    summary(model, (3, 32, 32))
    
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