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

from src.data import download_data, load_data

from utils.types import DataMode


def handle_model_training(subset_labels, data_mode, num_samples_per_class):
    download_data(subset_labels, data_mode, num_samples_per_class) # download data if not already
    load_data(subset_labels, data_mode, num_samples_per_class) # grab and parse .npy files (one file for each class)