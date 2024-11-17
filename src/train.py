import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data import download_stroke_data, download_img_data, load_simplified_data, load_stroke_data, test_display_img
from src.models import cnn, tcn, gan, rnn
from utils.image_processing import vector_to_raster, full_strokes_to_vector_images
from utils.types import DataMode, ModelType


def handle_model_training(subset_labels, data_mode, num_samples_per_class, model_type):
    # map label names to idxs
    label_map = {label: i for i, label in enumerate(subset_labels)}

    if data_mode == DataMode.FULL:
        download_stroke_data(subset_labels, data_mode, num_samples_per_class) # download strokes from full dataset
        X, y = load_stroke_data(subset_labels, data_mode, num_samples_per_class) # grab and parse .npy files (one file for each class)

    else:
        download_img_data(subset_labels, data_mode, num_samples_per_class) # download data if not already
        X, y = load_simplified_data(subset_labels, data_mode, num_samples_per_class) # grab and parse .npy files (one file for each class)

    # y is always the class label as an index following the label map
    if data_mode == DataMode.FULL: # x is stroke data with temporal aspect
        train_generator(X, y, subset_labels, model_type)

    # if data_mode == simplified: x is 255x255 final image data
    if data_mode == DataMode.SIMPLIFIED: # x is 255x255 final image data
        train_classifier(X, y, subset_labels, num_samples_per_class, 255)

    if data_mode == DataMode.REDUCED: # x is 28x28 final image data
        train_classifier(X, y, subset_labels, num_samples_per_class, 28)
    
    
def train_generator(X, y, subset_labels, model_type):
    # ensure data loaded properly by inspecting an image or two
    rand_idxs = np.random.randint(0, X.shape[0], 10) 
    for rand_idx in rand_idxs:
        X_vec = full_strokes_to_vector_images(X[rand_idx])
        in_size = int(np.max([np.max(stroke) for stroke in X_vec if len(stroke) > 0]))
        test_img = vector_to_raster([X_vec], in_size=in_size, out_size=256, line_diameter=8, padding=4)[0]
        test_display_img(test_img, subset_labels[y[rand_idx]], rand_idx)

    if model_type == ModelType.TCN:
        tcn.train_tcn(X, y)        

    if model_type == ModelType.RNN:
        rnn.train_rnn(X, y) 

    if model_type == ModelType.GAN:
        gan.train_gan(X, y) 

    else:
        print("incorrect modeltype specified for training generation")

def train_classifier(X, y, subset_labels, num_samples_per_class, img_size):  
    # ensure data loaded properly by inspecting an image or two or 10
    rand_idxs = np.random.randint(0, X.shape[0], 10)    
    for rand_idx in rand_idxs:
        test_display_img(X[rand_idx], subset_labels[y[rand_idx]], rand_idx)

    cnn.train_cnn(X, y)