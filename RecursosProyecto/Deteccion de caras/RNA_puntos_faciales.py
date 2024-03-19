import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_feather('resultado_aumentado.feather')

X = df['Image'].apply(lambda x: np.array(x.split(), dtype=float))
y = df.drop('Image', axis=1).values

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir las imágenes a matrices 2D
X_train = X_train.apply(lambda x: x.reshape(96, 96))
X_test = X_test.apply(lambda x: x.reshape(96, 96))

# Convertir las matrices 2D a tensores 3D
X_train = np.stack(X_train.values, axis=0)[:, :, :, np.newaxis]
X_test = np.stack(X_test.values, axis=0)[:, :, :, np.newaxis]

# Crear el modelo
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(30)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluar el modelo
model.evaluate(X_test, y_test)

# Guardar el modelo
model.save('modelo_puntos_faciales.h5')
