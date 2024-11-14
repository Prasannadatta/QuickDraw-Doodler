# QuickDraw-Doodler
Final Project for ECS 271 Machine Learning and Discovery at UC Davis. Project will be able to generate learned doodles in the way humans are able to, as well as classify various doodles from other people.

## sample command to download data
- Download and prepare to train 20,000 simplified samples (only final 28x28 doodle images)
`python main.py -m train -dm simplified -nspc 20000`
`python main.py --mode train --data_mode simplified --num_samples_per_class 20000`

- Download and prepare to train 5,000 full samples (includes the doodle's path and images are 256x256)
`python main.py -m train -dm full -nspc 5000`
`python main.py --mode train --data_mode full --num_samples_per_class 5000`

## Setup notes
- Train code would be ideal on gpu but should be designed to work on both
- DataMode is an enum variable that will represent one of 3 options:
    1. **full**: Used for the generator, this comes from the raw dataset and once downloaded will contain the x, y, and t values of strokes taken for doodles.
    These will be numpy arrs, and for each sample we access them like dictionaries.
    ```
    ex. for a given class, get second stroke's y value array of the 5th sample:
    X[4]['y'][1]

    sample stroke data:
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
    ```
    **Probably needs to be normalized, but not sure the best way for that yet.**

    2. **Simplified**: Data originally comes invector coordinate x, y pairs, and the time values are removed. When saving the .npy the vectors are rasterized into 2D arrays representing the 255x255 images of the doodles. Normalized pixel values between 0 and 1.

    3. **Reduced**: Same as simplified, but the dimensions of the image are reduced down to 28x28 and antialiasing is applied to better represent the original image.

- `config/subset_classes.json` is where we specify the list of classes used for each of the 3 modes, since using all 354 classes would be insane for a class project. 

## Things to try:
- Deep convnet is probably sufficient for training classifier,
    - Resnet type architecture may be worth investigating

- For Generations:
    - Google made sketchRNN, might be good starting point
    - GANs good for generation
    - Temporal convolution networks (TCNs) can retain spatial and temporal aspects of data