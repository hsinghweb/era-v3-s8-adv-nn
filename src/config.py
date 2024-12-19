import torch

class Config:
    # Dataset
    DATA_ROOT = "./data"
    NUM_CLASSES = 10
    
    # Training
    BATCH_SIZE = 128
    EPOCHS = 24
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model
    MODEL_SAVE_PATH = "model.pth" 
    
    # Add OneCycleLR parameters
    ONE_CYCLE_LR = True
    MAX_LR = 0.01
    DIV_FACTOR = 25
    PCT_START = 0.3