import torch

class Config:
    # Dataset
    DATA_ROOT = "./data"
    NUM_CLASSES = 10
    
    # Training
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model
    MODEL_SAVE_PATH = "model.pth" 