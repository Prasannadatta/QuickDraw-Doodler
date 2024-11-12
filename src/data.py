from quickdraw import QuickDrawData, QuickDrawDataGroup

import os
from itertools import islice
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from utils.types import DataMode

data_dir = "quickdraw_data"  # Directory to save the data
os.makedirs(data_dir, exist_ok=True)

def list_all_classes():
    qdd = QuickDrawData()
    print(qdd.drawing_names)

def render_image_data(strokes, out_img_size):
    """
    Render stroke data to an image of the specified size.

    Args:
        strokes (list): List of strokes, where each stroke is a tuple of two lists (x_coords, y_coords).
        out_img_size (tuple): Size of the output image, either (28, 28) or (256, 256).

    Returns:
        np.ndarray: A numpy array representing the grayscale image.
    """
    # create a blank image with the specified size
    image = Image.new("L", out_img_size, "white")
    draw = ImageDraw.Draw(image)
    
    scaling_factor = out_img_size[0] / 256  # Scale coordinates from 256x256 space to target size
    
    for x_coords, y_coords in strokes:
        # scale coordinates to fit within the target image size
        scaled_x = [int(x * scaling_factor) for x in x_coords]
        scaled_y = [int(y * scaling_factor) for y in y_coords]
        
        # draw the lines for each stroke
        coordinates = list(zip(scaled_x, scaled_y))
        draw.line(coordinates, fill="black", width=1)
    
    # return img as np array
    return np.array(image, dtype=np.uint8)

def test_display_img(img, label):
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

def download_data(subset_labels, data_mode, num_samples_per_class, data_dir="quickdraw_data/"):
    """
    Download a specified number of samples for each class and save as .npy files.
    Checks if already download before with the given number of samples and data mode.

    Args:
        classes (list of str): List of classes to download.
        data_mode (DataMode): Mode for downloading data ('full' or 'simplified').
        num_samples (int): Number of samples to download per class.
        data_dir (str): File path
    """
    data_dir += data_mode.value
    os.makedirs(data_dir, exist_ok=True)

    for label in subset_labels:
        label_path = os.path.join(data_dir, f"{label}.npy") # combine label and data directory to make data file path

        # check if data path already exists
        if os.path.exists(label_path):
            
            # if data path exists, check if it's the same number of samples currently passed in
            preexisting_data = np.load(label_path, allow_pickle=True).item()
            if preexisting_data['images'].shape[0] == num_samples_per_class:
                print(f"Found {num_samples_per_class} {data_mode.value} doodles for {label} already downloaded.")
                continue
            else:
                print(f"Found {preexisting_data.shape[0]} {data_mode.value} ddodles for {label}, but num_samples given is {num_samples_per_class}, updating download...")

        else:
            print(f"No {data_mode.value} samples for {label}, downloading now...")

        # initialize group for current label
        qddg = QuickDrawDataGroup(label)
        samples = list(islice(qddg.drawings, num_samples_per_class))

        image_data_dim = (num_samples_per_class, 28,28) if data_mode == DataMode.SIMPLIFIED else (num_samples_per_class, 256,256)

        # doodles are 28x28 pixels if simplified, else 256x256
        images = np.empty(image_data_dim, dtype=np.uint8) 

        # only collect data on strokes if full data mode not simplified
        strokes = np.empty(num_samples_per_class, dtype=object) if data_mode == DataMode.FULL else None

        for i, drawing in enumerate(samples):
            images[i] = render_image_data(drawing.image_data, image_data_dim[1:])
            if data_mode == DataMode.FULL:
                strokes[i] = drawing.strokes

        # put into dict
        if data_mode == DataMode.SIMPLIFIED:
            data_dict = {'images': images}
        else:
            data_dict = {'images': images, 'strokes': strokes}

        # save data to appropriate dir based on label name and data mode
        np.save(label_path, data_dict)
        print(label_path)

        print(f"**SAVED {num_samples_per_class} of {data_mode.value} {label} samples.")

def load_data(subset_labels, data_mode, num_samples_per_class, data_dir="quickdraw_data/"):
    """
    Load and prepare data from .npy files for the specified classes.

    Args:
        classes (list of str): List of class names to load (e.g., ['cat', 'dog']).
        data_mode (DataMode): Mode for using data ('full' or 'simplified').

    Returns:
        dict: A dictionary containing prepared 'images' and optionally 'strokes' if in full mode.
    """
    print("**LOADING DATA FROM .npy FILES")
    total_samples = num_samples_per_class * len(subset_labels)
    
    # image dimensions based on datamode
    image_data_dim = (total_samples, 28,28) if data_mode == DataMode.SIMPLIFIED else (total_samples, 256,256)
    
    # allocate arrays for images/labels and optionally for strokes
    images = np.empty(image_data_dim, dtype=np.float32)
    labels = np.empty(total_samples, dtype=np.uint8)
    strokes = np.empty(total_samples, dtype=object) if data_mode == DataMode.FULL else None

    for i, class_name in enumerate(subset_labels):
        # get file path and load .npy file
        cur_data_dir = f"{data_dir}{data_mode.value}/{class_name}.npy"
        data = np.load(cur_data_dir, allow_pickle=True).item()
        
        # slice indices for this class
        start_idx = i * num_samples_per_class
        end_idx = start_idx + num_samples_per_class

        # load images and normalize pixel values to [0, 1] range
        images[start_idx:end_idx] = data['images'][:num_samples_per_class] / 255.0
        labels[start_idx:end_idx] = i # label images by index

        # if in full mode load strokes as well
        if data_mode == DataMode.FULL and 'strokes' in data:
            strokes[start_idx:end_idx] = data['strokes'][:num_samples_per_class]

    # data dict
    prepared_data = {'images': images, 'labels': labels}
    if data_mode == DataMode.FULL:
        prepared_data['strokes'] = strokes

    print(f"Loaded and prepared {total_samples} images with labels for model training (data mode: {data_mode})")
    return prepared_data