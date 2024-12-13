import argparse
import numpy as np
import json
import torch

from src.classify import handle_doodle_classification
from src.generate import handle_doodle_generation
from src.train import handle_model_training
from utils.enum_types import DataMode, ModelType, Mode

# set seed globally to apply everywhere
np.random.seed(123)

def main():
    parser = argparse.ArgumentParser(description="Choose method for using QuickDraw Doodler: inferring, generating, or training on drawings")

    # operation mode
    parser.add_argument(
        '-m', '--mode', 
        type=Mode,
        choices=list(Mode), 
        default=Mode.TRAIN,
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
        default=10000,
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

    # path to trained model for generating/inferring
    parser.add_argument(
        '-mp', '--model_path',
        type=str,
        help="Path to trained model for generation or classification."
    )

    parser.add_argument(
        '-pdp', '--processed_data_path',
        type=str,
        default=None,
        help="Optional to specify when training if you have ran the training code where data is downloaded and preprocessed in the dataloader"
    )
    args = parser.parse_args()

    # check model_path arg
    if args.mode in [Mode.GENERATE, Mode.CLASSIFY] and not args.model_path:
        parser.error(f"Argument --model_path is required when mode is {args.mode.value}")

    # of the available 354 classes for training we will start with as a subset for testing
    with open("config/subset_classes.json", 'r') as subset_file:
        subset_dict = json.load(subset_file)
        print(subset_dict)
        subset_labels = subset_dict[args.data_mode.value]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.mode == Mode.CLASSIFY:
        handle_doodle_classification(args.model_type, args.model_path, device, subset_labels)

    if args.mode == Mode.GENERATE:
        handle_doodle_generation(args.model_type, args.model_path, device, args.label)

    if args.mode == Mode.TRAIN:
        handle_model_training(subset_labels, args.data_mode, args.num_samples_per_class, args.model_type, device, args.processed_data_path)

if __name__ == '__main__':
    main()