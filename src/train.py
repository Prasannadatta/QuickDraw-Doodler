import numpy as np
from tqdm import tqdm
import json

from utils.get_data import *
from utils.process_data import SequentialStrokeData, test_display_img
from models import cnn, rnn
from utils.image_rendering import vector_to_raster, full_strokes_to_vector_images, animate_strokes
from utils.enum_types import DataMode, ModelType


def handle_model_training(subset_labels, data_mode, num_samples_per_class, model_type, device):
    with open("config/model_params.json", 'r') as config_file:
        model_configs = json.load(config_file)
    
    # map label names to idxs
    label_map = {label: i for i, label in enumerate(subset_labels)}

    if data_mode == DataMode.FULL:
        download_stroke_data(subset_labels, data_mode, num_samples_per_class) # download strokes from full dataset
        X, y = load_stroke_data(subset_labels, data_mode, num_samples_per_class) # grab and parse .npy files (one file for each class)

    else:
        download_img_data(subset_labels, data_mode, num_samples_per_class) # download data if not already
        X, y = load_simplified_data(subset_labels, data_mode, num_samples_per_class) # grab and parse .npy files (one file for each class)

    # y is always the class label as an index following the label map
    if data_mode == DataMode.FULL: # x is stroke data with temporal aspect
        train_generator(X, y, subset_labels, model_type, device, model_configs)

    # if data_mode == simplified: x is 255x255 final image data
    if data_mode == DataMode.SIMPLIFIED: # x is 255x255 final image data
        train_classifier(X, y, subset_labels, num_samples_per_class, 255, model_configs)

    if data_mode == DataMode.REDUCED: # x is 28x28 final image data
        train_classifier(X, y, subset_labels, num_samples_per_class, 28, model_configs)
    
    
def train_generator(X, y, subset_labels, model_type, device, model_configs): 
    # ensure data loaded properly by inspecting an image or two
    print("Normalizing stroke data...")

    print("Rasterizing samples...")
    rand_idxs = np.random.randint(0, X.shape[0], len(subset_labels))
    for rand_idx in rand_idxs:
        X_vec = full_strokes_to_vector_images(X[rand_idx])
        in_size = int(np.max([np.max(stroke) for stroke in X_vec if len(stroke) > 0]))
        test_img = vector_to_raster([X_vec], in_size=in_size, out_size=256, line_diameter=8, padding=4)[0]
        test_display_img(test_img, subset_labels[y[rand_idx]], rand_idx)
        break       

    if model_type == ModelType.RNN:
        rnn_configs = model_configs['rnn']
        print("Normalizing stroke data...")
        #animate_strokes(Xnorm[rand_idxs[0]], use_actual_time=False, save_gif=True, gif_fp="output/doodle_anims/const_time.gif")

        dataset = SequentialStrokeData(
            X,
            y,
            rnn_configs['max_seq_len'],
            random_scale_factor=rnn_configs['random_scale_factor'],
            augment_stroke_prob=rnn_configs['augment_stroke_prob']
        )
        rnn.train_rnn(dataset, subset_labels, device, rnn_configs)

    else:
        print("incorrect modeltype specified for training generation")

def train_classifier(X, y, subset_labels, num_samples_per_class, img_size):  
    # ensure data loaded properly by inspecting an image or two or 10
    rand_idxs = np.random.randint(0, X.shape[0], 10)    
    for rand_idx in rand_idxs:
        test_display_img(X[rand_idx], subset_labels[y[rand_idx]], rand_idx)

    cnn.train_cnn(X, y)