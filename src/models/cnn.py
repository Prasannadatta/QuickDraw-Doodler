# Model
# def train_cnn(X, y):
#     pass
import torch
import torch.nn as nn

# main model
# i=image size should be given in parameter
class classifierCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(QuickDrawCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
    
    accuracy = 100 * correct / len(train_loader.dataset)
    return total_loss / len(train_loader), accuracy

# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    
    accuracy = 100 * correct / len(val_loader.dataset)
    return total_loss / len(val_loader), accuracy


////////////////////////
# handle train function
    # Hyperparameters
num_epochs = 10

# Instantiate the model, loss function, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = classifierCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load test dataset
test_dataset = datasets.ImageFolder(root='PATH_TO_TEST_DATA', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Track performance
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


test_accuracy = test_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")
////////////////////////

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    
    accuracy = 100 * correct / len(test_loader.dataset)
    return accuracy

# test_accuracy = test_model(model, test_loader, device)
# print(f"Test Accuracy: {test_accuracy:.2f}%")

