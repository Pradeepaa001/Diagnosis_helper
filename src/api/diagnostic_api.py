from models.model_loader import load_model
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load trained model once
model = load_model()

def predict_image(image_path):
    """Predicts disease from medical image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return "Disease Detected" if prediction == 1 else "No Disease"
