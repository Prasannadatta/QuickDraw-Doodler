import torch
import os
import json
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import re

from models.vae import DoodleGenRNN
from utils.metrics_visualize import distribution_comparison
from utils.process_data import SequentialStrokeData, get_real_samples_from_dataloader, init_sequential_dataloaders_from_dataset
from utils.image_rendering import animate_strokes

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

def load_data(preprocessed_datapath):
    with open("config/model_params.json", 'r') as config_file:
        rnn_configs = json.load(config_file)['rnn']

    data_files = os.listdir(preprocessed_datapath)
    data_file = [f for f in data_files if 'test' in f][0] 
    data_fp = os.path.join(preprocessed_datapath, data_file)
    print(f"Found following preprocessed data files to load: {data_fp}")
    test_dataset = SequentialStrokeData(preprocessed_fp=data_fp)
    test_loader = init_sequential_dataloaders_from_dataset(test_dataset, rnn_configs, shuffle=False)

    return test_loader, rnn_configs

def handle_doodle_generation(model_fp, device, preprocessed_datapath, sample_out_dir="output/generated", seq_len=169):
    if not os.path.isfile(model_fp) and model_fp != "all":
        raise FileNotFoundError(f"Error: File not found at path specified: {model_fp}")

    test_loader, rnn_configs = load_data(preprocessed_datapath)
    spatial_scale_factor = test_loader.dataset.dxy_std
    temporal_scale_factor = test_loader.dataset.dt_std
    model_class = DoodleGenRNN
    
    # if model_fp == all, loop over all models found in ckpt dir
    if model_fp == 'all':
        ckpt_dir = "output/model_ckpts/"
        fps = os.listdir(ckpt_dir)
        model_paths = [os.path.join(ckpt_dir, fp) for fp in fps if fp.endswith(".pt") and 'cnn' not in fp.lower()]
    # else model paths will just be the path passed in
    else:
        model_paths = [model_fp]

    for model_path in model_paths:
        try:
            vae, _, _ = load_model(model_path, model_class, device)
            print(f"Succesfully loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model checkpoint: {model_path}\nlikely old architecture not compatible with saved dataset: {e}")
            torch.cuda.empty_cache()
            continue

        model_fn = os.path.splitext(os.path.basename(model_path))[0]
        sample_out_fp = f"{sample_out_dir}/{model_fn}-doodle.gif"

        try:
            sketch = vae.sample_sketch(seq_len=seq_len)
        except Exception as e:
            print(f"Error sampling from model checkpoint: {model_path}\nShitty old model idk what else to say")
            del vae
            torch.cuda.empty_cache()
            continue
        
        animate_strokes(sketch.numpy(), use_actual_time=True, save_gif=True, gif_fp=sample_out_fp, dxy_std=spatial_scale_factor, dt_std=temporal_scale_factor)
        evaluate_vae_dists(vae, test_loader, rnn_configs, model_path)
        torch.cuda.empty_cache()
        del vae

def evaluate_vae_dists(
        vae,
        test_loader,
        rnn_config,
        model_fp,
        metrics_out_dir="output/generated",
    ):

    seq_pt_count = 0
    sample_count = 0
    max_seq_points = 5000
    gen_x, gen_y, gen_t = [], [], []
    while seq_pt_count < max_seq_points:
        gen_sketch = vae.sample_sketch(rnn_config['max_seq_len'])
        
        seq_pt_count += gen_sketch.size(0)
        sample_count += 1
        gen_dx = gen_sketch[:, 0].numpy().flatten()
        gen_dy = gen_sketch[:, 1].numpy().flatten()
        gen_dt = gen_sketch[:, 2].numpy().flatten()
        gen_x.append(gen_dx)
        gen_y.append(gen_dy)
        gen_t.append(gen_dt)
    gen_x = np.concatenate(gen_x)[:max_seq_points]
    gen_y = np.concatenate(gen_y)[:max_seq_points]
    gen_t = np.concatenate(gen_t)[:max_seq_points]

    real_x, real_y, real_t = get_real_samples_from_dataloader(test_loader, max_samples=max_seq_points)
    epoch = int(re.search(r"epoch(\d+)-", model_fp).group(1))

    fig, ax = plt.subplots(3,1, figsize=(16, 12))
    fig.suptitle("Generator Metrics Over Epochs", fontsize=16)
    jsds, wds = [], []
    for i, (real_data, gen_data, data_var) in enumerate([(real_x, gen_x, 'dx'), (real_y, gen_y, 'dy'), (real_t, gen_t, 'dt')]):
        fig, ax, jsd, wd = distribution_comparison(fig, ax, i, real_data, gen_data, data_var, epoch, metrics_out_dir)
        jsds.append(round(jsd, 4))
        wds.append(round(wd, 4))

    model_fn = os.path.splitext(os.path.basename(model_fp))[0]
    fname = f"{metrics_out_dir}/{model_fn}-dists.png"
    fig.savefig(fname, dpi=300) 
    print(f"Distribution metrics saved to {fname}")
    plt.close(fig)
