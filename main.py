import argparse
import numpy as np

from src.infer import handle_doodle_inferring
from src.generate import handle_doodle_generation
from src.train import handle_model_training

from utils.types import DataMode

# 10 of the available 300 classes for training we will start with as a subset for testing, maybe more if desired.
subset_labels = ["cat", "dog", "car", "tree", "house", "flower", "fish", "star", "apple", "face"]

# set seed globally to apply everywhere
np.random.seed(123)

def main():
    parser = argparse.ArgumentParser(description="Choose method for using QuickDraw Doodler: inferring, generating, or training on drawings")

    # operation mode
    parser.add_argument(
        '--mode', 
        choices=['infer', 'generate', 'train'], 
        required=True, 
        help="Choose the operation mode: 'infer' to perform inference on a drawing, 'generate' to create a new drawing, or 'train' to train on drawing data"
    )

    # data mode for training
    parser.add_argument(
        '--data_mode',
        type=DataMode,
        choices=list(DataMode),
        default=DataMode.FULL,
        help="Specify data mode for training: 'full' to include strokes, or 'simplified' for images only (default: full)"
    )

    # number of samples to get/use per class
    parser.add_argument(
        '--num_samples_per_class',
        type=int,
        default=2000,
        help="Specify how many samples to use for training."
    )

    args = parser.parse_args()

    if args.mode == 'infer':
        handle_doodle_inferring()

    if args.mode == 'generate':
        handle_doodle_generation()

    if args.mode == 'train':
        handle_model_training(subset_labels, args.data_mode, args.num_samples_per_class)

if __name__ == '__main__':
    main()