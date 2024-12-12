import torch
import os

from torch.optim import Adam

from models.vae import DoodleGenRNN
from utils.enum_types import ModelType
from utils.process_data import local_normalize_stroke_data, test_display_img
from utils.image_rendering import vector_to_raster, full_strokes_to_vector_images, animate_strokes
from utils.get_data import *

def load_model(model_fp, model_class, device, label):
    ckpt = torch.load(model_fp, map_location=device)
    gen_model = model_class(**ckpt['hyperparams'])
    gen_model.load_state_dict(ckpt['model_state_dict'])

    # make sure label passed in to condition model on was a label it was trained on
    if label and label not in gen_model.subset_labels:
        raise Exception(f"Error: Label: {label} is not one that {model_fp} was trained on. Either change the label one of: '{gen_model.subset_labels}' or generate unconditionally.")

    gen_model.to(device)

    optim = Adam(gen_model.parameters())
    optim.load_state_dict(ckpt['optimizer_state_dict'])

    metadata = {
        'train_loss': ckpt['train_loss'],
        'val_loss': ckpt['val_loss'],
        'hyperparams': ckpt['hyperparams']
    }

    return gen_model, optim, metadata

def generate_conditional(rnn, label, device, seq_len=125):
    # input random latent vector
    z = torch.randn(1, rnn.latent_size).to(device)
    x, y, t = 0, 0, 0 # init absolute positions
    strokes = []
    label_tensor = torch.tensor([label], device=device)

    with torch.no_grad():
        # decode latent vec into mdn params
        output = rnn.decode(z, seq_len, label_tensor, None, None)

        # get mixture coefficients of mixture density network
        rnn.mdn.get_mixture_coeff(output)

        # sample strokes for each time step from Gaussian mixture density
        for time_step in range(seq_len):
            dx, dy, dt, p = rnn.mdn.sample_mdn(time_step)
            x += dx
            y += dy
            t += dt
            strokes.append([x, y, t, p])
            
    return np.array(strokes)

def generate_uncoditional(model, T=1, z_scale=1):
    pass

def handle_doodle_generation(model_type, model_fp, device, label=None, sample_out_fp="output/doodle_anims/sampled_doodle.gif"):
    if not os.path.isfile(model_fp):
        raise FileNotFoundError(f"Error: File not found at path specified: {model_fp}")
    
    if model_type == ModelType.RNN:
        rnn, optim, metadata = load_model(model_fp, DoodleGenRNN, device, label)
        if label:
            label_map = {label: i for i, label in enumerate(rnn.subset_labels)}
            label = label_map[label]
            sample = generate_conditional(rnn, label, device)
        else:
            sample = generate_uncoditional(rnn, device)
    print(f"Loaded model stats:{metadata}")
    print(sample)
    animate_strokes(sample, use_actual_time=True, save_gif=True, gif_fp=sample_out_fp)
    
    print(f"Saved sample animation to: {sample_out_fp}")