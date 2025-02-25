import torch
import torchvision.models as models

def load_model():
    model = models.resnet18(pretrained=True)  # Using ResNet for medical images
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Assume binary classification (Healthy/Diseased)
    return model
