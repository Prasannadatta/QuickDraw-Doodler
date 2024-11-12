import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data import download_data, load_data, test_display_img

from utils.types import DataMode


def handle_model_training(subset_labels, data_mode, num_samples_per_class):
    download_data(subset_labels, data_mode, num_samples_per_class) # download data if not already
    data_dict = load_data(subset_labels, data_mode, num_samples_per_class) # grab and parse .npy files (one file for each class)

    # x images are final doodle outputs, y labels are idx of subset labels, opt x strokes are if using full data to train generator
    x_images, y = data_dict['images'], data_dict['labels']
    if data_mode == DataMode.FULL:
        x_strokes = data_dict['strokes']

    # labels are loaded as idxs, for reference we can make a hashmap to relate label idxs to actual labels
    label_map = {i: label for i, label in enumerate(subset_labels)}
    print(label_map)

    # ensure data loaded properly by inspecting an image or two
    rand_idx = np.random.randint(0, num_samples_per_class * len(subset_labels))
    test_display_img(x_images[rand_idx], label_map[y[rand_idx]])