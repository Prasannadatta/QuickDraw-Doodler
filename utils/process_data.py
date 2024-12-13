import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from rdp import rdp
from utils.image_rendering import animate_strokes

import matplotlib.pyplot as plt
import numpy as np
import os

class SequentialStrokeData(Dataset):
    def __init__( # if strokes are none, preprocessed_fp must contain filepath to data
            self,
            strokes=None,   # list of np arrs dims (x, y, t, p) x N; (will be transposed to Nx4)
            labels=None,    # array of labels (same length as strokes)
            max_len=169,    # max sequence len
            random_scale_factor=0.0,    # amt to randomly scale stokes
            augment_stroke_prob=0.0,    # chance to augment strokes
            limit=1000,
            preprocessed_fp=None      # filepath to data that's been saved by this class
        ):

        if strokes is None and preprocessed_fp is None:
            raise ValueError("ERROR: Must specify either stroke data or filepath to preprocessed data")

        # load preprocessed data if fp given
        if preprocessed_fp is not None:
            self.load_preprocessed(preprocessed_fp)
        else:
            self.max_len = max_len
            self.random_scale_factor = random_scale_factor
            self.augment_stroke_prob = augment_stroke_prob
            self.limit = limit

            print("Preprocessing data...")
            self.strokes, self.sort_idx = self.preprocess(strokes) # list of np arrs dims N x (x, y, t, p)

            if labels is not None:
                self.labels = torch.tensor(labels, dtype=torch.long)[self.sort_idx]
            else:
                self.labels = None
                
            # Compute global normalization stats (x,y,t)
            print("Computing normalization stats...")
            self.x_mean, self.x_std, self.y_mean, self.y_std, self.t_mean, self.t_std = self.calculate_global_stats()

    def preprocess(self, strokes):
        """
        Transpose to shape (N,4), filter long sequences, convert to tensors, clamp values, sort by sequence lens
        """
        raw_data = []
        seq_len = []
        count_data = 0

        # Renamed stroke to stroke_seq for clarity 
        for stroke_seq in strokes:
            if stroke_seq.shape[0] == 4: # if 4xN, transpose to Nx4
                stroke_seq = stroke_seq.T
            # RDP - Ramer-Douglas-Peuker algorithm
            Ramer_set = []

            split_ind = np.where(stroke_seq[:, 3] == 0)[0]

            split_stroke = np.split(stroke_seq, split_ind + 1) 

            # define epsilon (Test: 2.0, 1.0, .5)
            # Test conclusive: epsilon of 1.0 cuts data substantially while maintaining clear lines
            epsilon = 1.0

            for stroke in split_stroke:
                xy_points = stroke[:, :2]
                mask = rdp(xy_points, epsilon=epsilon, algo='iter', return_mask=True)
                Ramer_set.append(stroke[mask])

            Ramer_set = np.concatenate(Ramer_set, axis=0)

            # animate_strokes(stroke_seq, delta=False, use_actual_time=False, save_gif=True, num_frames=500, gif_fp= "output/doodle_anims/RDPep=1.0.gif")
        
            # only take strokes less than hp arg for max length
            if len(Ramer_set) <= self.max_len:
                Ramer_set = to_tensor(Ramer_set) # convert stroke np arr to tensor
                Ramer_set[:,:2].clamp_(-self.limit, self.limit)  # clamp x,y (inplace)
                raw_data.append(Ramer_set)
                seq_len.append(len(Ramer_set))
                count_data += 1

        sort_idx = np.argsort(seq_len)
        processed_strokes = [raw_data[ix] for ix in sort_idx]
        print(f"total drawings <= max_seq_len is {count_data}")
        return processed_strokes, sort_idx
    
    def calculate_global_stats(self):
        # Concatenate all for mean/std
        # shape (N,4): x,y,t,p
        all_data = torch.cat(self.strokes, dim=0)
        x = all_data[:,0]
        y = all_data[:,1]
        t = all_data[:,2]

        x_mean, x_std = x.mean(), x.std()
        y_mean, y_std = y.mean(), y.std()
        t_mean, t_std = t.mean(), t.std()

        print(x_mean, x_std, y_mean, y_std, t_mean, t_std)

        return x_mean, x_std, y_mean, y_std, t_mean, t_std

    def save_preprocessed(self, filepath):
        """
        Save the preprocessed data to a file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'constructor_params': {
                'max_len': self.max_len,
                'random_scale_factor': self.random_scale_factor,
                'augment_stroke_prob': self.augment_stroke_prob,
                'limit': self.limit
            },
            'preprocessed_data': {
                'strokes': self.strokes,
                'sort_idx': self.sort_idx,
                'labels': self.labels,
                'x_mean': self.x_mean,
                'x_std': self.x_std,
                'y_mean': self.y_mean,
                'y_std': self.y_std,
                't_mean': self.t_mean,
                't_std': self.t_std
            }
        }
        torch.save(data, filepath)
        print(f"Preprocessed data and constructor parameters saved to '{filepath}'.")

    def load_preprocessed(self, filepath):
        """
        Load preprocessed data and constructor parameters from a file.
        """
        data = torch.load(filepath)
        
        # Load constructor parameters
        constructor_params = data.get('constructor_params', {})
        self.max_len = constructor_params.get('max_len', 169)
        self.random_scale_factor = constructor_params.get('random_scale_factor', 0.0)
        self.augment_stroke_prob = constructor_params.get('augment_stroke_prob', 0.0)
        self.limit = constructor_params.get('limit', 1000)
        
        # Load preprocessed data
        preprocessed_data = data.get('preprocessed_data', {})
        self.strokes = preprocessed_data.get('strokes', [])
        self.sort_idx = preprocessed_data.get('sort_idx', [])
        self.labels = preprocessed_data.get('labels', None)
        self.x_mean = preprocessed_data.get('x_mean', torch.tensor(0.0))
        self.x_std = preprocessed_data.get('x_std', torch.tensor(1.0))
        self.y_mean = preprocessed_data.get('y_mean', torch.tensor(0.0))
        self.y_std = preprocessed_data.get('y_std', torch.tensor(1.0))
        self.t_mean = preprocessed_data.get('t_mean', torch.tensor(0.0))
        self.t_std = preprocessed_data.get('t_std', torch.tensor(1.0))
        
        print(f"Preprocessed data and constructor parameters loaded from '{filepath}'.")

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        data = self.strokes[idx].clone() # (N,4): x,y,t,p

        # Optional augmentation before normalization and delta:
        if self.augment_stroke_prob > 0:
            data = random_augment(data, self.augment_stroke_prob)

        # Normalize (x,y,t) using global stats from above
        data[:,0] = (data[:,0] - self.x_mean)/self.x_std
        data[:,1] = (data[:,1] - self.y_mean)/self.y_std
        data[:,2] = (data[:,2] - self.t_mean)/self.t_std

        # Compute deltas: dx, dy, dt
        dx = torch.cat([data[:1,0], data[1:,0]-data[:-1,0]])
        dy = torch.cat([data[:1,1], data[1:,1]-data[:-1,1]])
        dt = torch.cat([data[:1,2], data[1:,2]-data[:-1,2]])

        # Pen state one-hot:
        # data[:,3] = p. p=1 pen down, p=0 pen up.
        p = data[:,3]
        p1 = (p==1).float()
        p2 = (p==0).float()
        p3 = torch.zeros_like(p) # will use for EOS later

        # Combine into stroke-6 format: [dx, dy, dt, p1, p2, p3]
        # paper uses stroke 5 originally, since they don't use t
        stroke_6 = torch.stack([dx, dy, dt, p1, p2, p3], dim=1) # (N,6)

        # Add EOS token at the end
        eos = torch.tensor([0,0,0,0,0,1], dtype=stroke_6.dtype).unsqueeze(0)
        stroke_6 = torch.cat([stroke_6, eos], dim=0)

        # Add SOS token at the start
        sos = torch.zeros(1,6, dtype=stroke_6.dtype)
        stroke_6 = torch.cat([sos, stroke_6], dim=0) # (N+2,6)

        # Optional random scaling after delta computation:
        if self.random_scale_factor > 0:
            stroke_6 = random_scale(stroke_6, self.random_scale_factor)
        if self.augment_stroke_prob > 0:
            stroke_6 = random_augment(stroke_6, self.augment_stroke_prob)

        label = self.labels[idx] if self.labels is not None else None
        length = stroke_6.shape[0]

        # Return (data, length, label) similar to what you'd do for a collate_fn
        return (stroke_6, length, label)

def get_max_seq_len(sequences):
    max_len = -1
    for seq in sequences:
        cur_seq_len = len(seq)
        if len(seq) > max_len:
            max_len = cur_seq_len

    return max_len

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        raise Exception('Input must be a tensor or ndarray.')

def random_scale(data, factor):
    """
    Augment data by stretching x and y axis randomly [1-e, 1+e].
       Here 'data' will be in stroke-6 format:
       data shape: (N,6) = [dx, dy, dt, p1, p2, p3]
       But scaling only applies to dx, dy. We ignore dt and pen states.
    """
    data = data.clone()
    x_scale = (torch.rand(()) - 0.5) * 2 * factor + 1.0
    y_scale = (torch.rand(()) - 0.5) * 2 * factor + 1.0
    data[:,0] *= x_scale  # dx
    data[:,1] *= y_scale  # dy
    return data

def random_augment(data, prob):
    """
    Perform data augmentation by randomly dropping out strokes before delta computation."""
    data = data.clone()
    result = []

    # Keep track of last stroke
    prev_stroke = [data[0,0].item(), data[0,1].item(), data[0,3].item()] # (x,y,p)
    count = 0
    stroke = [data[0,0].item(), data[0,1].item(), data[0,3].item()]
    # We'll store t in parallel arrays and reattach later
    # Actually, let's store full line and then handle at the end
    # We'll need to keep time dimension consistent:
    augmented = [data[0].clone().tolist()]

    for i in range(1, len(data)):
        candidate = data[i].clone().tolist()  # [x,y,t,p]
        p_current = candidate[3]
        p_prev = prev_stroke[2]
        if p_current == 1 or p_prev == 1:
            count = 0
        else:
            count += 1
        check = p_current == 0 and p_prev == 0 and count > 2
        if check and (torch.rand(()) < prob):
            # merge candidate into stroke
            augmented[-1][0] += candidate[0]
            augmented[-1][1] += candidate[1]
            # t: we can either merge by taking the max t or sum.
            # In original logic, we are merging stroke coordinates,
            # For t, let's just keep the later time as it's cumulative.
            augmented[-1][2] = max(augmented[-1][2], candidate[2])
        else:
            augmented.append(candidate)
            prev_stroke = [candidate[0], candidate[1], candidate[3]]

    return torch.tensor(augmented, dtype=torch.float)

def pad_batch(sequences, max_len):
    """
    Pad the batch to fixed length.
    """
    batch_size = len(sequences)
    dim = sequences[0].size(1) # should be 6
    output = torch.zeros(batch_size, max_len, dim)
    for i in range(batch_size):
        seq = sequences[i]
        l = seq.size(0)
        if l > max_len:
            # if sequence is longer than max_len, we truncate (rare if carefully chosen max_len)
            l = max_len
        output[i,:l,:] = seq[:l,:]
        if l < max_len:
            #set last col to 1
            output[i, l:, -1] = 1
    return output

class CollateFn:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        return collate_sketches(batch, self.max_seq_len)

def collate_sketches(batch, max_len=250):
    # batch: list of (stroke_6, length, label)
    batch.sort(key=lambda x: x[1], reverse=True)
    data, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    if labels[0] is not None:
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = None

    #torch.set_printoptions(edgeitems=2)
    #print("Here is the end of a sample before padding: \n", data[-2:])
    padded_samples = pad_batch(data, max_len)
    #print("Here is the end of a sample after padding: \n", padded_samples[-2:])

    return padded_samples, lengths, labels

def init_sequential_dataloaders_from_numpy(X, y, config, num_workers=4):
    """
    Take in data in numpy normalized format and:
    1. strat split data
    2. init sequential datasets
    3. init dataloaders
    """
    # shuffle and split data while ensuring balance class distribution
    Xtrain, Xeval, ytrain, yeval = train_test_split(X, y, test_size=0.2, stratify=y)
    Xval, Xtest, yval, ytest = train_test_split(Xeval, yeval, test_size=0.1, stratify=yeval)

    # custom datasets
    train_dataset = SequentialStrokeData(
        Xtrain,
        ytrain,
        max_len=config['max_seq_len'],
        random_scale_factor=config['random_scale_factor'],
        augment_stroke_prob=config['augment_stroke_prob']
    )
    val_dataset = SequentialStrokeData(
        Xval,
        yval,
        max_len=config['max_seq_len']
    )
    test_dataset = SequentialStrokeData(
        Xtest,
        ytest,
        max_len=config['max_seq_len']
    )

    train_loader = init_sequential_dataloaders_from_dataset(train_dataset, config, shuffle=True, num_workers=num_workers)
    val_loader = init_sequential_dataloaders_from_dataset(val_dataset, config, shuffle=False, num_workers=num_workers)
    test_loader = init_sequential_dataloaders_from_dataset(test_dataset, config, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def init_sequential_dataloaders_from_dataset(dataset, config, shuffle=False, num_workers=4):
    # collate fn needs max_len arg, but can't pass arg directly to it in Dataloader
    # use class that returns collate fn with this arg
    collate = CollateFn(config['max_seq_len'])

    # dataloaders
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=True
    )

    return loader

def get_real_samples_from_dataloader(loader, max_samples=2000):
    real_x, real_y, real_t = [], [], []
    count = 0
    for batch in loader:
        Xbatch, seq_lens, _ = batch
        # Xbatch (B, L, 6) = [dx, dy, dt, p1, p2, p3]
        # Flatten and extract dx, dy
        dx = Xbatch[..., 0].numpy().flatten()
        dy = Xbatch[..., 1].numpy().flatten()
        dt = Xbatch[..., 2].numpy().flatten()
        
        real_x.append(dx)
        real_y.append(dy)
        real_t.append(dt)
        count += len(dx)
        if count >= max_samples:
            break
    
    real_x = np.concatenate(real_x)[:max_samples]
    real_y = np.concatenate(real_y)[:max_samples]
    real_t = np.concatenate(real_t)[:max_samples]
    return real_x, real_y, real_t

def local_normalize_stroke_data(data):
    '''
    Old method of normalizing pen stroke data, normalized within samples instead of globally
    '''
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
    '''
    probably doesn't work anymore since changing data methods,
    but don't really need it so not fixing it til i do
    '''
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