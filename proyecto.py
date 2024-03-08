import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random
from scipy import ndimage
import tensorflow_hub as hub

emociones = pd.read_csv("RecursosProyecto/datasets_proyectos/emociones/icml_face_data.csv", delimiter=",")
emociones.columns = emociones.columns.str.strip()

emociones = emociones.iloc[1:21]
emociones.pixels = emociones.pixels.apply(lambda x: np.array(x.split()).reshape(48,48).astype(float))/255


def transformation(x):
    rotate_rand = random.randint(-30,30)
    x = ndimage.rotate(x, rotate_rand, reshape = False)

    bright_rand = random.uniform(1.5, 2)
    x = np.clip(x * bright_rand, 0.0, 255.0)

    return x

emociones.pixels = emociones.pixels.apply(lambda x: cv2.resize(x, (96,96)))
emociones_augmented = emociones.copy()

for n in range(len(emociones_augmented.pixels)):
    emociones_augmented.pixels.iloc[n] = transformation(emociones.pixels.iloc[n])


result = pd.concat([emociones, emociones_augmented])

# plt.imshow(result.iloc[10].pixels, cmap='gray')
# plt.show()
# plt.imshow(result.iloc[30].pixels, cmap='gray')
# plt.show()

resnet_url = "https://www.kaggle.com/models/tensorflow/resnet-50/frameworks/TensorFlow2/variations/feature-vector/versions/1"
IMAGE_SHAPE = (96, 96)
COUNT_UNIQUE_EMOTIONS = len(pd.unique(result.emotion))

feature_extractor_layer = hub.KerasLayer(resnet_url,
                                           trainable=False, # congela los patrones subyacentes
                                           name='feature_extraction_layer',
                                           input_shape=IMAGE_SHAPE+(1,)) # define la forma de la imagen de entrada

model = tf.keras.Sequential([
    feature_extractor_layer, # utiliza la capa de extracción de características como la base
    tf.keras.layers.Dense(COUNT_UNIQUE_EMOTIONS, activation='softmax', name='output_layer') # crea nuestra propia capa de salida
])


