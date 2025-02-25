import os
import torch

def load_model(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found.")
    return model
