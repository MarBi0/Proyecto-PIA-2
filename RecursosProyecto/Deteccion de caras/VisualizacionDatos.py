import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lee el CSV
df = pd.read_csv('../datasets_proyectos/puntos_faciales/data.csv')

# Itera sobre las filas del DataFrame
for index, row in df.iterrows():
    # Obtiene las coordenadas x e y del punto
    x = row['left_eye_center_x']
    y = row['left_eye_center_y']

    # Obtiene la representación de la imagen y la convierte en un array de valores
    image_values = np.fromstring(row['Image'], sep=' ', dtype=int)

    # Realiza el reshape de la imagen a 94x94
    image_array = image_values.reshape(96, 96)  # Cambié 94 a 96 porque el reshape es de 96x96

    # Muestra la imagen
    plt.imshow(image_array, cmap='gray')

    # Agrega un punto rojo en las coordenadas
    plt.scatter(x, y, color='red', marker='o')

    # Ajusta el aspecto de la imagen
    plt.axis('off')

    # Muestra la imagen con el punto rojo
    plt.show()
