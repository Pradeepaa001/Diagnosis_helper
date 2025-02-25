import torch
import torch.nn as nn
import torch.optim as optim
from model_loader import load_model
from api.data_loader import load_medical_data

# Load dataset
train_loader, val_loader = load_medical_data("data/")

# Load model
model = load_model()

# Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(epochs=10):
    """Trains the model on the medical dataset."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "models/diagnostic_model.pth")

if __name__ == "__main__":
    train_model()
