import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchinfo import summary

from tqdm import tqdm
import numpy as np

from src.process_data import init_sequential_dataloaders
    
class DoodleRNN(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(DoodleRNN, self).__init()

        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True, # data is loaded with batch dim first (b, seq_len, points)
            bidirectional=True
        )

        # fc layer to classify lstm outputs
        self.fc = nn.Linear(hidden_size * 2, num_classes) # bidirectional means * 2 hidden_size for num in neurons

    def forward(self, x, lens):
        # pack pad the sequence for length uniformity
        # enforce sorted since we sort in desc seq len order in data loader
        packed_in = pack_padded_sequence(x, lens.cpu(),batch_first=True, enforce_sorted=True)
        packed_out, (hn, cn) = self.lstm(packed_in)

def train_rnn(X, y, subset_labels, device, batch_size=32):
    train_loader, val_loader, test_loader = init_sequential_dataloaders(X, y, batch_size)

    rnn = DoodleRNN(
        in_size=4, # 4 features -> dx, dy, dt, p
        hidden_size=128, # num features in lstm hidden state
        num_layers=4, # num of stacked lstm layers
        num_classes=len(subset_labels), # num unique classes in dataset
        dropout=0.2 # chance of neurons to dropout (turn to 0)
    )