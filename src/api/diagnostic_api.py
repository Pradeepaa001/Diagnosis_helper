import torch
import torchvision.transforms as transforms
from PIL import Image
from src.models.model_loader import load_model

# Load pre-trained model
model = load_model()

# Define preprocessing steps
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Function to make predictions
def diagnose(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    return probabilities.numpy()
