import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import os

class SequentialStrokeData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        # length of sequences used later for padding
        self.sequence_lens = np.array([sample.shape[0] for sample in data])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]  # (n_points, 4)
        label = self.labels[idx]
        length = sample.shape[0]

        # convert to torch tensor from numpy arr before returning
        return torch.from_numpy(sample).float(), length, label

def sequential_collate_fn(batch):
    # sequences must be in descending order by length for pack padded sequence
    # x[1] corresponds to len of sequence in dataloader
    batch.sort(key=lambda x: x[1], reverse=True)

    # unpack batch (list of tuples)
    samples, lengths, labels = zip(*batch)

    # convert lengths to tensor
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # pad sequences of variable length
    # batch dimension remains at front (b, max_seq_len, points)
    padded_samples = pad_sequence(samples, batch_first=True)

    return padded_samples, lengths, labels

def init_sequential_dataloaders(X, y, batch_size):
    """
    Take in data in numpy normalized format and:
    1. strat split data
    2. init sequential datasets
    3. init dataloaders
    """
    # shuffle and split data while ensuring balance class distribution
    Xtrain, Xeval, ytrain, yeval = train_test_split(X, y, test_size=0.3, stratify=y)
    Xval, Xtest, yval, ytest = train_test_split(Xeval, yeval, test_size=0.3, stratify=yeval)

    # custom datasets
    train_dataset = SequentialStrokeData(Xtrain, ytrain)
    val_dataset = SequentialStrokeData(Xval, yval)
    test_dataset = SequentialStrokeData(Xtest, ytest)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sequential_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=sequential_collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=sequential_collate_fn, num_workers=4)

    return train_loader, val_loader, test_loader

def local_normalize_stroke_data(data):
    norm_data = np.empty(data.shape[0], dtype=object)
    stats = []  # List to store min and max values for x, y, and t for each sample
    for i, sample in enumerate(data):
        x, y, t, p = sample
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        total_t = t[-1] -t[0]
        if total_t == 0:
            total_t = 1e-6

        # edge cases if the min ever equals max (vert or hor lines)
        x_range = x_max - x_min if x_max - x_min != 0 else 1e-6
        y_range = y_max - y_min if y_max - y_min != 0 else 1e-6

        # normalize x and y to edges of bbox (xmax and ymax)
        # x, y are normalized before finding deltas to ensure scale consistency with bbox
        # t normalized before deltas to ensure temproal dynamics are relative to other strokes, not other drawings
        x_norm = (x - x_min) / x_range
        y_norm = (y - y_min) / y_range
        t_norm = (t - t[0]) / total_t

        # absolute values not necessary to process sequential inputs
        dx = np.diff(x_norm, prepend=x_norm[0]).astype(np.float32)
        dy = np.diff(y_norm, prepend=y_norm[0]).astype(np.float32)
        dt = np.diff(t_norm, prepend=t_norm[0]).astype(np.float32)

        # each row will be a timestep where all features are grouped together
        norm_data[i] = np.stack([dx, dy, dt, p], axis=1)

        stats.append({'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 't_min': t[0], 't_max': total_t})

    return norm_data, stats

def unnormalize_stroke_data(norm_data, stats):
    unnorm_data = np.empty(norm_data.shape[0], dtype=object)
    for i, sample in enumerate(norm_data):
        dx, dy, dt, p = sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3]
        x_min, x_max = stats[i]['x_min'], stats[i]['x_max']
        y_min, y_max = stats[i]['y_min'], stats[i]['y_max']
        t_min, t_max = stats[i]['t_min'], stats[i]['t_max']

        x_range = x_max - x_min if x_max - x_min != 0 else 1e-6
        y_range = y_max - y_min if y_max - y_min != 0 else 1e-6
        t_range = t_max - t_min if t_max - t_min != 0 else 1e-6

        # reconstruct normalized positions from deltas
        x_norm = np.cumsum(dx)
        y_norm = np.cumsum(dy)
        t_norm = np.cumsum(dt)

        # unnormalize the positions
        x = x_norm * x_range + x_min
        y = y_norm * y_range + y_min
        t = t_norm * t_range + t_min

        unnorm_data[i] = np.stack([x, y, t, p], axis=0)
        
    return unnorm_data

def test_display_img(img, label, idx):
    os.makedirs("output/sample_outputs/", exist_ok=True)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.savefig(f"output/sample_outputs/{label}-{idx}.png")
    #plt.show()