import torch
import os

from torch.optim import Adam

from models.vae import DoodleGenRNN
from utils.enum_types import ModelType
from utils.process_data import local_normalize_stroke_data, test_display_img
from utils.image_rendering import vector_to_raster, full_strokes_to_vector_images, animate_strokes
from utils.get_data import *

def load_model(model_fp, model_class, device):
    ckpt = torch.load(model_fp, map_location=device)
    gen_model = model_class(**ckpt['hyperparams'])
    gen_model.load_state_dict(ckpt['model_state_dict'])

    gen_model.to(device)

    optim = Adam(gen_model.parameters())
    optim.load_state_dict(ckpt['optimizer_state_dict'])

    metadata = {
        'train_loss': ckpt['train_loss'],
        'val_loss': ckpt['val_loss'],
        'hyperparams': ckpt['hyperparams']
    }

    return gen_model, optim, metadata

def handle_doodle_generation(model_fp, device, sample_out_fp="output/doodle_anims/sampled_doodle.gif", seq_len=169):
    if not os.path.isfile(model_fp):
        raise FileNotFoundError(f"Error: File not found at path specified: {model_fp}")

    model_class = DoodleGenRNN
    vae, _, _ = load_model(model_fp, model_class, device)

    sketch = vae.sample_sketch(seq_len=seq_len)