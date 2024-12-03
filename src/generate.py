import torch
import os

from torch.optim import Adam

from src.enum_types import ModelType
from src.models.rnn import DoodleGenRNN
from src.process_data import local_normalize_stroke_data, test_display_img
from src.image_rendering import vector_to_raster, full_strokes_to_vector_images, animate_strokes
from src.get_data import *

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

    return gen_model, optim

def generate_conditional(model, label):
    pass

def generate_uncoditional(model, T=1, z_scale=1):
    pass

def handle_doodle_generation(model_type, model_fp, device, label=None):
    if not os.path.isfile(model_fp):
        raise FileNotFoundError(f"Error: File not found at path specified: {model_fp}")
    
    if model_type == ModelType.RNN:
        rnn = load_model(model_fp, DoodleGenRNN, device, label)
        if label:
            label_map = {label: i for i, label in enumerate(rnn.subset_labels)}
            label = label_map[label]
            generate_conditional(rnn, label)
        else:
            generate_uncoditional(rnn)
        



        