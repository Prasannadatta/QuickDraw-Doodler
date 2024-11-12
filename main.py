import argparse

from src.infer import train_model
from src.generate import train_model
from src.model import train_model

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
        pass

    if args.mode == 'generate':
        pass

    if args.mode == 'train':
        train_model()