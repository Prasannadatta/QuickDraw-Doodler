import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

def handle_doodle_classification(image_path, device):
    """
    Classifies a doodle image into one of the predefined label subsets.

    Parameters:
        image_path (str): Path to the doodle image to classify.
        device (torch.device): Device (CPU/GPU) on which the model is loaded.

    Returns:
        str: Predicted label.
    """
    # Paths to model checkpoint and label subset
    classifier_model = "trained_models/classifierCNN_epoch12_20241208-201712.pt"
    label_subset_path = "label_subset.txt"  # Replace with the actual path to your label subset file

    # Step 1: Load the model
    if not os.path.exists(classifier_model):
        raise FileNotFoundError(f"Model checkpoint not found at {classifier_model}")
    
    model = load_model(classifier_model, device)

    # Step 2: Load the label subset or may be just get that array
    if not os.path.exists(label_subset_path):
        raise FileNotFoundError(f"Label subset file not found at {label_subset_path}")
    
    with open(label_subset_path, "r") as file:
        label_subset = [line.strip() for line in file.readlines()]

    # Step 3: Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to match the model's input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
    ])
    
    # Step 4: Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')  # Open the image
        input_tensor = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Step 5: Perform the prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(input_tensor)  # Get raw predictions from the model
        _, predicted_idx = torch.max(outputs, 1)  # Get index of the max logit value
    
    # Step 6: Get the predicted label
    predicted_label = label_subset[predicted_idx.item()]

    # Step 7: Display the image with the predicted label
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()

    return predicted_label

#ask brandon how to?
def load_model(checkpoint_path, device):
    """
    Loads the model from a checkpoint.

    Parameters:
        checkpoint_path (str): Path to the model checkpoint.
        device (torch.device): Device (CPU/GPU) to load the model.

    Returns:
        torch.nn.Module: Loaded model.
    """
    # Define the CNN model architecture
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = torch.nn.Linear(64 * 5 * 5, 128)
            self.fc2 = torch.nn.Linear(128, len(label_subset))

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Instantiate and load model
    model = SimpleCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
