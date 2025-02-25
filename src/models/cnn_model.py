import torch
import torch.nn as nn
import torch.optim as optim

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MedicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 112 * 112, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def train_model(dataloader, num_classes=2, epochs=5, lr=0.001):
    model = MedicalCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), "data/cnn_model.pth")
    print("Model training complete. Saved as cnn_model.pth")
