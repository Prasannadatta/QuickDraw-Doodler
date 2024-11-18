import argparse
import numpy as np
import json

from src.infer import handle_doodle_inferring
from src.generate import handle_doodle_generation
from src.train import handle_model_training

from utils.types import DataMode, ModelType

# set seed globally to apply everywhere
np.random.seed(123)

def main():
    parser = argparse.ArgumentParser(description="Choose method for using QuickDraw Doodler: inferring, generating, or training on drawings")

    # operation mode
    parser.add_argument(
        '-m', '--mode', 
        choices=['infer', 'generate', 'train'], 
        required=True, 
        help="Choose the operation mode: 'infer' to perform inference on a drawing, 'generate' to create a new drawing, or 'train' to train on drawing data"
    )

    # data mode for training
    parser.add_argument(
        '-dm', '--data_mode',
        type=DataMode,
        choices=list(DataMode),
        default=DataMode.FULL,
        help="Specify data mode for training: 'full' to include strokes, or 'simplified' for images only (default: full)"
    )

    # number of samples to get/use per class
    parser.add_argument(
        '-nspc', '--num_samples_per_class',
        type=int,
        default=2000,
        help="Specify how many samples to use for training."
    )

    # model selection
    parser.add_argument(
        '-mt', '--model_type',
        type=ModelType,
        choices=list(ModelType),
        default=ModelType.RNN,
        help="Choose the type of model to train, generate from, or infer on. "
    )

    args = parser.parse_args()

    # of the available 354 classes for training we will start with as a subset for testing
    with open("config/subset_classes.json", 'r') as subset_file:
        subset_dict = json.load(subset_file)
        print(subset_dict)
        subset_labels = subset_dict[args.data_mode.value]

    if args.mode == 'infer':
        handle_doodle_inferring()

    if args.mode == 'generate':
        handle_doodle_generation()

    if args.mode == 'train':
        handle_model_training(subset_labels, args.data_mode, args.num_samples_per_class, args.model_type)

if __name__ == '__main__':
    main()