import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from models.cnn import classifierCNN

#ask brandon how to?
def load_model(model_fp, device):
    # Instantiate and load model
    model = classifierCNN().to(device)
    checkpoint = torch.load(model_fp, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def handle_doodle_classification(model_type, model_fp, device, subset_labels, image_path="output/sample_outputs/apple-17730.png"):
    if not os.path.isfile(model_fp):
        raise FileNotFoundError(f"Error: File not found at path specified: {model_fp}")

    model = load_model(model_fp, device)

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
    predicted_label = subset_labels[predicted_idx.item()]

    # Step 7: Display the image with the predicted label
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()
    print(f"Prediction: {predicted_label}")

    return predicted_label
