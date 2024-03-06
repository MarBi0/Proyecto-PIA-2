import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

emociones = pd.read_csv("datasets_proyectos/emociones/icml_face_data.csv", delimiter=",")
emociones.columns = emociones.columns.str.strip()

emociones = emociones.iloc[1:21]
emociones.pixels = emociones.pixels.apply(lambda x: np.array(x.split()).reshape(48,48).astype(float))/255

print(emociones.pixels.iloc[0].ndim)

print(emociones.iloc[0].pixels.shape)

plt.imshow(emociones.iloc[0].pixels, cmap='gray')
plt.show()

emociones.pixels = emociones.pixels.apply(lambda x: (tf.image.decode_image(x, channels=2)))
# train_datagen_augmented = ImageDataGenerator(rescale=1*2)
#print(train_datagen_augmented)




# rotation_range = 20,
# shear_range = 0.2,
# zoom_range = 0.2,
# width_shift_range = 0.2,
# height_shift_range = 0.2,
# horizontal_flip = True
