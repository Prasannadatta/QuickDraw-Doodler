from enum import Enum

class Mode(Enum):
    TRAIN = 'train'
    GENERATE = 'generate'
    CLASSIFY = 'classify'

class DataMode(Enum):
    FULL = 'full' # includes strokes taken to complete drawing
    SIMPLIFIED = 'simplified' # only contains final output of drawing
    REDUCED = 'reduced' # final output only and shrinked to 28x28 (MNIST size)

class ModelType(Enum):
    RNN = 'rnn' # baseline comparison using google's pretrained RNN
    TCN = 'tcn' # Use our implemented Temporal Convolutional Network
    GAN = 'gan' # Train using a generative advesarial network
    CNN = 'cnn' # Only used for classifying simplified output images