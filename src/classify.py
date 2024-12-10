import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from models.cnn import classifierCNN
import numpy as np

def preprocess_image_for_cnn(image_path, device):
    try:
        # Load the image and convert to grayscale
        image = Image.open(image_path).convert('L')  # Convert to grayscale (1 channel)
        
        # Resize the image to (28, 28)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert image to a numpy array
        image_array = np.array(image).astype(np.float32)  # Convert to float32
        
        # Normalize pixel values to the range [0, 1]
        image_array /= 255.0
        
        # Add the channel dimension (for grayscale)
        image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 28, 28)
        
        # Add the batch dimension
        image_tensor = torch.tensor(image_array).unsqueeze(0).to(device)  # Shape: (1, 1, 28, 28)
        
        return image_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def load_model(model_fp, device):
    # Instantiate and load model
    model = classifierCNN().to(device)
    checkpoint = torch.load(model_fp, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def handle_doodle_classification(model_type, model_fp, device, subset_labels, image_path="output/sample_outputs/tree-23766.png"):
    if not os.path.isfile(model_fp):
        raise FileNotFoundError(f"Error: File not found at path specified: {model_fp}")

    model = load_model(model_fp, device)

    # Step: Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')  # Open the image
        input_tensor = preprocess_image_for_cnn(image_path, device)  # Apply transformations and add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Step: Perform the prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(input_tensor)  # Get raw predictions from the model
        _, predicted_idx = torch.max(outputs, 1)  # Get index of the max logit value
    
    # Step: Get the predicted label
    predicted_label = subset_labels[predicted_idx.item()]

    # Step: Display the image with the predicted label
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()
    print(f"Prediction: {predicted_label}")

    return predicted_label
