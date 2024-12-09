# Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from datetime import datetime
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from src.process_data import init_sequential_dataloaders
from utils.metrics_visualize import plot_cnn_metrics, log_metrics
from tqdm import tqdm
from torchinfo import summary

# main model
# image size should be given in parameter
class classifierCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(classifierCNN, self).__init__()
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
    model.train()
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

        # Metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    accuracy = 100 * correct / len(train_loader.dataset)
    return total_loss / len(train_loader), accuracy

# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / len(val_loader.dataset)
    return total_loss / len(val_loader), accuracy

# test function
def test_model(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    confusion = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Extract metrics
    accuracy = 100 * np.trace(confusion) / np.sum(confusion)  # Overall accuracy
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # Save the plot
    fn = f"ClassifierCNN_test_{cur_time}.png"
    fp = os.path.join('output/classifier_test_loop/', fn)
    plt.savefig(fp, dpi=400)
    print(f"Test plot saved at: {fp}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": confusion,
        "classification_report": report
    }

def train_cnn(X, y, device, image_size, model_configs, subset_labels):
    # Extract configurations
    batch_size = model_configs['batch_size']
    num_epochs = model_configs['num_epochs']
    learning_rate = model_configs['learning_rate']
    class_names = subset_labels

    # Reshape X to include channel dimension
    X = X.reshape(-1, 1, image_size, image_size)  # Ensure shape is (N, 1, 28, 28)
    print(f"Reshaped X shape: {X.shape}")  # Debugging: Check new shape

    # Move data to device
    X, y = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.long).to(device)

    print(f"Shape of X: {X.shape} ----------------------------------")  # Expect (20000, 1, 28, 28) for grayscale
    print(f"Shape of y: {y.shape} ----------------------------------")  # Expect (20000,)    

    # Prepare dataset and dataloaders
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = classifierCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics storage
    metrics = {
        'train': {'loss': [], 'accuracy': []},
        'val': {'loss': [], 'accuracy': []}
    }

    input_size = (1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    # Generate the model summary
    model_summary = summary(
        model,
        input_size=input_size,  # Single input example
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=1
    )
    print(model_summary)


    # Training loop with tqdm
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Logging
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save metrics
        metrics['train']['loss'].append(train_loss)
        metrics['train']['accuracy'].append(train_accuracy)
        metrics['val']['loss'].append(val_loss)
        metrics['val']['accuracy'].append(val_accuracy)

        # Save model per epoch
        cur_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        model_fp = "output/classifier_model/"
        model_fn = f"classifierCNN_epoch{epoch+1}_{cur_time}"
        os.makedirs(model_fp, exist_ok=True)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, model_fp + model_fn + '.pt')

        # Create and save logs
        with open(model_fp + model_fn + '.log', 'w', encoding='utf-8') as model_summary_file:
            model_summary_file.write(str(model_summary))

        # Metrics visualization and logging
        log_dir = "output/classifier_model_metrics/"
        plot_cnn_metrics(metrics, cur_time, epoch + 1, log_dir)
        log_metrics(metrics, cur_time, epoch + 1, log_dir)
    
    # Final Test Metrics
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_metrics = test_model(model, test_loader, device, class_names)

    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Precision: {test_metrics['precision']:.2f}")
    print(f"Recall: {test_metrics['recall']:.2f}")
    print(f"F1 Score: {test_metrics['f1_score']:.2f}")

    return model, metrics, test_metrics