import argparse

from src.infer import handle_doodle_inferring
from src.generate import handle_doodle_generation
from src.train import handle_model_training

def get_args():
    parser = argparse.ArgumentParser(description="Choose method for using QuickDraw Doodler: inferring, generating, or training on drawings")

    # operation mode
    parser.add_argument(
        '--mode', 
        choices=['infer', 'generate', 'train'], 
        required=True, 
        help="Choose the operation mode: 'infer' to perform inference on a drawing, 'generate' to create a new drawing, or 'train' to train on drawing data"
    )

    args = parser.parse_args()

    if args.mode == 'infer':
        handle_doodle_inferring()

    if args.mode == 'generate':
        handle_doodle_generation()

    if args.mode == 'train':
        handle_model_training()