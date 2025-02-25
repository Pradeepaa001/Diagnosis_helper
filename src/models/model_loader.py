import torch
import torch.nn as nn
import torchvision.models as models

def load_model(model_path="models/diagnostic_model.pth"):
    """Loads a pre-trained CNN model for medical diagnosis."""
    model = models.resnet18(pretrained=False)  # Change model as needed
    model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming binary classification

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model
