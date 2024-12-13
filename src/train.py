import numpy as np
from tqdm import tqdm
import json

from utils.get_data import *
from utils.process_data import SequentialStrokeData, init_sequential_dataloaders_from_dataset, init_sequential_dataloaders_from_numpy, test_display_img
from models import cnn, vae
from utils.image_rendering import vector_to_raster, full_strokes_to_vector_images, animate_strokes
from utils.enum_types import DataMode, ModelType


def handle_model_training(subset_labels, data_mode, num_samples_per_class, model_type, device, processed_data_path=None):
    with open("config/model_params.json", 'r') as config_file:
        model_configs = json.load(config_file)

    rnn_configs = model_configs['rnn']
    print(processed_data_path)
    
    # map label names to idxs
    label_map = {label: i for i, label in enumerate(subset_labels)}

    print("Gathering data...")
    if data_mode == DataMode.FULL:
        if processed_data_path is None:
            download_stroke_data(subset_labels, data_mode, num_samples_per_class) # download strokes from full dataset
            X, y = load_stroke_data(subset_labels, data_mode, num_samples_per_class) # grab and parse .npy files (one file for each class)
            # ensure data loaded properly by inspecting an image or two
            print("Rasterizing sample sketches...")
            rand_idxs = np.random.randint(0, X.shape[0], len(subset_labels))
            for rand_idx in rand_idxs:
                X_vec = full_strokes_to_vector_images(X[rand_idx])
                in_size = int(np.max([np.max(stroke) for stroke in X_vec if len(stroke) > 0]))
                test_img = vector_to_raster([X_vec], in_size=in_size, out_size=256, line_diameter=8, padding=4)[0]
                test_display_img(test_img, subset_labels[y[rand_idx]], rand_idx)
                break
            train_loader, val_loader, test_loader = init_sequential_dataloaders_from_numpy(X, y, rnn_configs, num_workers=4)
            
            # save datasets
            train_loader.dataset.save_preprocessed(f"quickdraw_data/vae_preprocessed-nspc{str(num_samples_per_class)[:-3]}k/train_vae_dataset.pt")
            val_loader.dataset.save_preprocessed(f"quickdraw_data/vae_preprocessed-nspc{str(num_samples_per_class)[:-3]}k/val_vae_dataset.pt")
            test_loader.dataset.save_preprocessed(f"quickdraw_data/vae_preprocessed-nspc{str(num_samples_per_class)[:-3]}k/test_vae_dataset.pt")
        else:
            # init datasets from filepath
            data_files = os.listdir(preprocessed_fp=processed_data_path)
            datasets = [SequentialStrokeData(fp) for fp in data_files]
            
            # sort datasets, since train will have most, val second, and third least
            # this is how we distinguish which dataset goes to which loader
            sorted_datasets = sorted(datasets, key= lambda dataset: dataset.__len__(), reverse=True)
            train_loader = init_sequential_dataloaders_from_dataset(sorted_datasets[0], rnn_configs, shuffle=True)
            val_loader = init_sequential_dataloaders_from_dataset(sorted_datasets[1], rnn_configs, shuffle=False)

    else:
        download_img_data(subset_labels, data_mode, num_samples_per_class) # download data if not already
        X, y = load_simplified_data(subset_labels, data_mode, num_samples_per_class) # grab and parse .npy files (one file for each class)

    # y is always the class label as an index following the label map
    if data_mode == DataMode.FULL: # x is stroke data with temporal aspect
        train_generator(train_loader, val_loader, subset_labels, model_type, device, rnn_configs)

    # if data_mode == simplified: x is 255x255 final image data
    if data_mode == DataMode.SIMPLIFIED: # x is 255x255 final image data
        train_classifier(X, y, subset_labels, num_samples_per_class, 255, model_configs)

    if data_mode == DataMode.REDUCED: # x is 28x28 final image data
        train_classifier(X, y, subset_labels, num_samples_per_class, 28, model_configs, device)
    
    
def train_generator(train_loader, val_loader, subset_labels, model_type, device, rnn_configs): 
    if model_type == ModelType.RNN:
        #animate_strokes(Xnorm[rand_idxs[0]], use_actual_time=False, save_gif=True, gif_fp="output/doodle_anims/const_time.gif")
        vae.train_rnn(train_loader, val_loader, subset_labels, device, rnn_configs)
    else:
        print("incorrect modeltype specified for training generation")

def train_classifier(X, y, subset_labels, num_samples_per_class, img_size, model_configs, device):  
    # ensure data loaded properly by inspecting an image or two or 10
    rand_idxs = np.random.randint(0, X.shape[0], 10)    
    for rand_idx in rand_idxs:
        test_display_img(X[rand_idx], subset_labels[y[rand_idx]], rand_idx)
    #changed the function name
    cnn.train_cnn(X, y, device, img_size, model_configs['cnn'], subset_labels)