from enum import Enum

class DataMode(Enum):
    FULL = 'full' # includes strokes taken to complete drawing
    SIMPLIFIED = 'simplified' # only contains final output of drawing
    REDUCED = 'reduced' # final output only and shrinked to 28x28 (MNIST size)