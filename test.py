import os
import glob
import math
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential,load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.python.keras.layers import Add, Input, Conv2D, Conv2DTranspose, Dense, Input, MaxPooling2D, UpSampling2D, Lambda


# 定議產生器
def data_generator(data_path, cale=3.0, target_size=(200, 200), batch_size=32, shuffle=True):
    x_lab = []
    imgs = img_to_array(load_img(data_path)).astype(np.uint8)
    arr_img = np.expand_dims(imgs, axis=0)
    x_lab.append(arr_img)
    np.stack(x_lab)
    x = np.concatenate([x for x in x_lab])
   
    yield x


def psnr(y_true, y_pred):
    return  -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path','-img_path', help='Input image path')

    return parser.parse_args()

def main():
    args = process_command()
    N_TRAIN_DATA = 1000
    N_TEST_DATA = 100
    BATCH_SIZE = 16
    test_x = next(
        data_generator(
            args.img_path,
            batch_size=N_TEST_DATA,
            shuffle=True
        )
    )
    model = load_model('my_model.h5',custom_objects={'psnr': psnr})

    pred = model.predict(test_x)
    img = array_to_img(test_x[0])
    img.save( "fileout.jpg", "JPEG" )

    plt.imshow(img)
    plt.axis('off')
    plt.show()
    


if __name__ == "__main__":
    main()