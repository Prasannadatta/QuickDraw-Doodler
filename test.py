from quickdraw import QuickDrawData, QuickDrawDataGroup
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from datasets import load_dataset, Features, ClassLabel, Value, Sequence
import os

from get_data import vector_to_raster

def test_display_img(img, label):
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

def test1():
    qddg = QuickDrawDataGroup('cat', recognized=True)
    for i in range(2):
        stroke_data = qddg.get_drawing().strokes
        print(stroke_data)
        img = vector_to_raster([stroke_data])
        print(img)
        img = img[0]

        test_display_img(img, 'cat')


        '''if isinstance(img, Image.Image):
            print(type(img), img.size)
        img_np = np.array(img, dtype=np.uint8)[:,:,0]
        print(img_np.shape)

        print(img_np[0])


        test_display_img(img_np, 'cat', 2)

        print(img_np.shape)
        reduced_img = cv2.resize(img_np, (28,28), interpolation=cv2.INTER_AREA)
        print(reduced_img)
        test_display_img(reduced_img, 'cat', 2)'''

def test2():
    #dataset = tfds.load()

    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/raw/{}.ndjson"

    subset_labels = ['cat','tree']
    
    total_fraction_of_data_to_download = (200) / 5e7
    cache_dir = ".hfcache"
    os.makedirs(cache_dir, exist_ok=True)
    label_data_dict = {}
    for label in subset_labels:
        iter_dataset = load_dataset(
            "json",
            data_files={label: base_url.format(label)},
            cache_dir=cache_dir,
            streaming=True
        )

        label_data_dict[label] = list(iter_dataset)
        print(len(list(iter_dataset)), list(iter_dataset)[0])
    print(label_data_dict)

if __name__ == '__main__':
    test2()
