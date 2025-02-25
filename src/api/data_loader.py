import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def load_medical_data(data_dir, batch_size=32):
    """Loads medical images dataset for training and validation."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    val_data = ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
