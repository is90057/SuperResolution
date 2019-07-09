import os
import glob
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.python.keras.layers import Add, Input, Conv2D, Conv2DTranspose, Dense, Input, MaxPooling2D, UpSampling2D, Lambda, Activation

from PIL import Image

# 產生低解析度的影像
def drop_resolution(x, scale=10.0):
    size = (x.shape[0], x.shape[1])
    small_size = (int(size[0]/scale), int(size[1]/scale))
    img = array_to_img(x)
    small_img = img.resize(small_size, 3)
    arr_img = img_to_array(small_img.resize(img.size,3))
    return arr_img

# 定議產生器
def data_generator(data_dir, mode, scale=3.0, target_size=(255, 255), batch_size=32, shuffle=True):
    for imgs in ImageDataGenerator().flow_from_directory(
        directory=data_dir,
        classes=[mode],
        class_mode=None,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=shuffle
    ):
        x = np.array([
            drop_resolution(img) for img in imgs
        ])
        yield x/255., imgs/255.

def psnr(y_true, y_pred):
    return  -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def main():
    DATA_DIR = '/home/is90057/Documents/train_data/'
    TEST_DATA_DIR = 'data/'
    N_TRAIN_DATA = 1000
    N_TEST_DATA = 100
    BATCH_SIZE = 16

    train_data_generator = data_generator(DATA_DIR, 'DIV2K_train_HR', batch_size=BATCH_SIZE)
    test_x, test_y = next(
        data_generator(
            TEST_DATA_DIR,
            'test',
            batch_size=N_TEST_DATA,
            shuffle=False
        )
    )
 
    inputs = Input((None, None, 3), dtype='float')
    level1_1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    level2_1 = Conv2D(64, (3,3), activation='relu', padding='same')(level1_1)

    level2_2 = Conv2DTranspose(64, (3,3), activation='relu', padding='same')(level2_1)
    level2 = Add()([level2_1, level2_2])

    level1_2 = Conv2DTranspose(64, (3,3), activation='relu', padding='same')(level2)
    level1 = Add()([level1_1, level1_2])

    decoded = Conv2D(3, (5, 5), activation='linear', padding='same')(level1)
    model = Model(inputs, decoded)
    model.summary()
    
    model.compile(
        loss='mean_squared_error', 
        optimizer='adam', 
        metrics=[psnr]
    )

    model.fit_generator(
        train_data_generator,
        validation_data=(test_x, test_y),
        steps_per_epoch=N_TRAIN_DATA//BATCH_SIZE,
        epochs=50
    )

    model.save('my_model.h5')
    pred = model.predict(test_x)


    img = array_to_img(test_x[0])
    img2 = array_to_img(pred[0])
    plt.figure("Image")
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()