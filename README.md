# QuickDraw-Doodler
Final Project for ECS 271 Machine Learning and Discovery at UC Davis. Project will be able to generate learned doodles in the way humans are able to, as well as classify various doodles from other people. Both using Google's [QuickDraw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset).

## Project Goal
Google has already trained a Sequence2Sequence Variational Autoencoder for this dataset. However they only fed the spatial dimensions and binary pen state of the stroke sequences (dx, dy, and p), even though the dataset recorded the time of the strokes and individual stroke progression (dt). The goal of this project is to incorporate the timing of these sketches. Capturing the temporal aspect as well as spatial. This allows us to not only model the pathing that humans draw sketches, but also learn the time variation between and within strokes. 


<p align="center">
  <img src="readme_figs/const_time.gif" width="300" alt="Visual for how Google's model receives the sketches">
  <img src="readme_figs/var_time.gif" width="300" alt="Visual for how our model receives the sketches">
</p>


These gifs are both from the same dataset (not generated). Here you can see how Google's model (left/top) would receive the data. Where all strokes are constant speeds, and therre is no pause in between strokes where a human would normally have to lift there pen and move it to continue the sketch. Our model (right/bottom) receives the pen strokes as a vector including the time deltas along with the spatial ones. Our model is designed to learn both of these aspects of how humans draw, and generate as such.

---

# Using The Project

## Preface
**1**: Project likely works on various python versions and operating systems, but only confirmed with the following:
  - Windows, Python 3.10.5 (Cuda tested)
  - Linux, Python 3.10.12 (Cuda not tested)

**2**: Model hyperparameters for VAE and CNN specified in `config/model_params.json`, and the choice of subset classes to use are defined in `config/subset_classes.json`

**3**: Model checkpoints are saved every 5 epochs, loss plots and metrics logs saved every epoch, and Distribution metrics saved every 20 epochs. Model is saved as `output/model_ckpts/DoodleGenRNN_epoch{X}_{Date-Time}.pt` where X is the current epoch at the time of saving, and Date-Time refers to the date and time the training started. Metrics and plots are saved the same way, but without the current epoch since they are simply updated with more current information each epoch. 

**4**: Model checkpoints are in the gitignore, but our chosen 'best' models are in the `trained_models/` directory. Models in this folder should be chosen when generating sequences or classifying sketches.

**5**: In the parameters below, specifying true for use layer norm and any value above 0 for recurrent dropout will use the custom implementation of an LSTM since the standard pytorch one does not include those. It is a good bit slower despite compiling with torch.jit.script(). Regular dropout hypeparameter will go into the standard pytorch LSTM, where it is applied between layers. This is only useful if num_lstm_layers > 1. 


## Package installation
- make a virtual environment
`python -m venv venv`

- activate the venv 
  - windows: `.\venv\Scripts\activate`
  - linux/mac: `source venv/bin/activate`

- install required packages
`pip install -r requirements.txt`

- optional: install pytorch for gpu use
  - install cuda and cudnn if not already installed
  - get the pip install cmd from pytorch website: https://pytorch.org/get-started/locally/
    - this cmd varies depending on os, python package installer, and cuda version

You are now ready to run the project using the instructions and arguments below

## CLI Arguments

Below are descriptions of the main command-line arguments for the `main.py` script.

- `-m, --mode`
  (Required, default: train)
  Specifies the operation mode. Choices are:
    * train: Download and prepare data, then train a model on the specified dataset.
    * generate: Use a trained model to generate new doodles. Requires `--model_path`.
    * infer: Perform inference/classification on input doodles. Requires `--model_path`.
  
- `-dm, --data_mode`
  (Default: full)
  Specifies the type of data used for training (and associated modes):
    * full: Full stroke-based data from the raw dataset (x, y, t sequences).
    * simplified: 255x255 rasterized doodle images (no timing data).
    * reduced: 28x28 downsampled and antialiased doodle images.
  
- `-nspc, --num_samples_per_class`
  (Default: 10000)
  Sets the number of samples to use per class for training. **Note** refers to total number of samples between train/val/test, where the split is 0.8/0.19/0.01
  
- `-mt, --model_type`
  (Default: RNN)
  Specifies the type of model architecture. Choices can include:
    * RNN: Recurrent neural network-based model (e.g., LSTM, GRU).
    * CNN: Convolutional neural network-based model (for image-based modes).
  
- `-mp, --model_path`
  Path to a saved, trained model checkpoint. Required when using `--mode generate` or `--mode infer`.


## Example Commands

- Download and prepare training data (2,000 samples per class) using simplified images:
`python main.py -m train -dm simplified -nspc 2000`
`python main.py --mode train --data_mode simplified --num_samples_per_class 10000`

- Train rnn for generation while downloading and processing data if missing or different amount that currently exists. (100,000 samples per class) using full stroke-based data:
`python main.py -m train -dm full -mt rnn -nspc 50000`
`python main.py --mode train --data_mode full --model_type rnn --num_samples_per_class 50000`

Once training is complete, you can generate new doodles or perform inference using trained models:
- Generate doodles with a trained model (model path required):
`python main.py -m generate -mt rnn -mp path/to/trained_model.pt`

- Classify/infer on a doodle with a trained model (model path required):
`python main.py -m infer -mt cnn -mp path/to/trained_model.pt`

## Data Structure
1. **full**: Used for the generator, this comes from the raw dataset and once downloaded will contain the x, y, and t values of strokes taken for doodles. These are converted into (sequence_len, 4) vectors and all normalized and converted to relative position/time (deltas) instead of global. Additionally we had a binary pen state where `0` indicates the pen is lifted (i.e. moving to new location without drawing) and `1` indicates pen is down and drawing.
    
    ```

    ORIGINAL sample stroke data:
    [ 
        [  // First stroke 
            [x0, x1, x2, x3, ...],
            [y0, y1, y2, y3, ...],
            [t0, t1, t2, t3, ...]
        ],
        [  // Second stroke
            [x0, x1, x2, x3, ...],
            [y0, y1, y2, y3, ...],
            [t0, t1, t2, t3, ...]
        ],
        ... // Additional strokes
    ]

    POST-PROCESSED sample stroke data:
    [
        [0      0      0      1],
        [dx1    dy1    dt1    p1],
        [dx2    dy2    dt2    p2],
        [dx3    dy3    dt3    p3],
        ...
        [dxn    dyn    dtn    pn]
    ]
    n = sequence length

    ```
    
2. **Simplified**: Data originally comes in vector coordinate x, y pairs, and the time values are removed. When saving the .npy the vectors are rasterized into 2D arrays representing the 255x255 images of the doodles. Normalized pixel values between 0 and 1.

3. **Reduced**: Same as simplified, but the dimensions of the image are reduced down to 28x28 and antialiasing is applied to better represent the original image.

- `config/subset_classes.json` is where we specify the list of classes used for each of the 3 modes, since using all 354 classes would be insane for a class project. 
- `config/model_params.json` is where we specify the dict of hyperparameters for the RNN and CNN. 


## Model Parameters

Located in `config/model_params.json` for both rnn and cnn
rnn params are as follows:

```

"rnn": {
    "in_size": 6,               # stroke_6 format (dx, dy, dt, p1, p2, p3)
    "num_epochs": 3,            # number of epochs for training
    "batch_size": 100,          # num samples per step
    "max_seq_len": 250,         # max length of sequences (of full samples not indv strokes) post RDP line simplification
    "latent_size": 128,         # size of the latent space between encoder and decoder
    "enc_hidden_size": 256,     # encoder size of each layers hidden and cell state vectors
    "dec_hidden_size": 512,     # decoder size of each layers hidden and cell state vectors
    "num_lstm_layers": 2,       # number of layers for the encoder and decoder LSTMs
    "recurrent_dropout": 0.1,   # fraction of recurrent connection nodes to drop to 0
    "dropout": 0.1,             # amount nodes input/output nodes to drop to 0
    "decoder_act": "leaky_relu",# activation fn to apply to decoder output
    "num_gmm_modes": 20,        # number of modes for the Gaussian Mixture Model
    "learning_rate": 0.001,     # initial learning rate for the optimizer, controls the size of gradient updates
    "lr_decay": 0.9999,         # exponential decay rate for learning rate per epoch (exponential decay)
    "kl_weight": 1,             # Weight applied to the KL divergence term in the loss function
    "kl_weight_start": 0.01,    # Initial value for KL weight; gradually increased during training
    "kl_tolerance": 0.25,        # Threshold for KL divergence at which it stops being penalized
    "kl_decay_rate": 0.99995,   # Rate at which the KL weight is increased during training
    "reg_covar": 1e-6,          # Regularization term for covariance in the GMM to prevent numerical instability
    "grad_clip": 1.0,           # Maximum value for gradient clipping to prevent exploding gradients
    "random_scale_factor": 0.1, # Maximum scale factor for randomly scaling strokes during augmentation
    "augment_stroke_prob": 0.08,# Probability of applying augmentation to a stroke during training
    "num_workers": 4            # Number of worker threads for data loading
}

```

---


## vae WIP

- [x] weight inits
  - [x] fc layer xavier-glorot
  - [x] lstm in to hidden xavier-glorot
  - [x] lstm hidden to hidden orthogonal

- [x] remove label conditions
  - (label args still remain for later updates)
- [x] remove attention bullshit
- [x] encoder pack padded sequences
- [x] handle SOS/EOS tokens in training
- [x] check kl loss
- [x] check recon loss
- [x] check mdn layers
  - fc layer may be missing
    - it wasn't, it was just in the vae not in the mdn
- [x] add correct recurrent dropout
- [x] layer norm - decoder?
- [x] check model_params.json alignment with vae args
- [x] sample from the model
