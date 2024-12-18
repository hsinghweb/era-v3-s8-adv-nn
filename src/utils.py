import torch
from tqdm import tqdm
from config import Config

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader):
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return running_loss / len(dataloader), accuracy 