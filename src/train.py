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

from data import gather_train_data, parse_train_data

def handle_model_training():
    pass