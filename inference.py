import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import os
from os import listdir
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Lambda, BatchNormalization
from tensorflow.keras import Model

IMG_W = 160
IMG_H = 80


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def get_emb_net(IMG_H, IMG_W):
    input_vec = Input((IMG_H, IMG_W, 1))
    x = BatchNormalization()(input_vec)
    x = Conv2D(4, (5, 5), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (5, 5), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(10, activation="relu")(x)

    x = BatchNormalization()(x)
    x = Dense(10, activation="relu")(x)

    embedding_network = Model(input_vec, x)
    return embedding_network


def get_model(IMG_H, IMG_W):
    embedding_network = get_emb_net(IMG_H, IMG_W)

    input_1 = Input((IMG_H, IMG_W, 1))
    input_2 = Input((IMG_H, IMG_W, 1))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = BatchNormalization()(merge_layer)
    output_layer = Dense(1, activation="sigmoid")(normal_layer)
    siamese_model = Model(inputs=[input_1, input_2], outputs=output_layer)

    return siamese_model


def get_loaded_model():

    weights_path = 'models/'

    best_weights_path = os.path.join(weights_path, "best_ckpt")

    loaded_siamese_model = get_model(IMG_H, IMG_W)
    loaded_siamese_model.load_weights(best_weights_path)
    return loaded_siamese_model


def preprocess_img(img):
    img = ImageOps.grayscale(img)
    img_arr = np.array(img)

    # invert image if background is black
    if img_arr.mean() > 220:
        img = ImageOps.invert(img)

    # crop image to bounding box
    img = img.crop(img.getbbox())

    # resize image to standard size
    img = img.resize((IMG_W, IMG_H))

    # binarize image to black and white values
    img_mean = np.array(img).mean()
    img = Image.eval(img, lambda p: 255 if p > img_mean else 0)
    return np.array(img)

#
# def get_img(url, img_name):
#   img = Image.open(url)
#   processed_img =
#
#   return {'img': img, 'of': img_name[:3], 'by': img_name[5:8]}
