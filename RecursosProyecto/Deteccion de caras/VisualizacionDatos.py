import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lee el CSV
df = pd.read_csv('../datasets_proyectos/puntos_faciales/data.csv')

# Itera sobre las filas del DataFrame
for index, row in df.iterrows():
    # Obtiene las coordenadas de los puntos de interés
    points = [
        ('left_eye_center_x', 'left_eye_center_y'),
        ('right_eye_center_x', 'right_eye_center_y'),
        ('left_eye_inner_corner_x', 'left_eye_inner_corner_y'),
        ('left_eye_outer_corner_x', 'left_eye_outer_corner_y'),
        ('right_eye_inner_corner_x', 'right_eye_inner_corner_y'),
        ('right_eye_outer_corner_x', 'right_eye_outer_corner_y'),
        ('left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y'),
        ('left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y'),
        ('right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y'),
        ('right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y'),
        ('nose_tip_x', 'nose_tip_y'),
        ('mouth_left_corner_x', 'mouth_left_corner_y'),
        ('mouth_right_corner_x', 'mouth_right_corner_y'),
        ('mouth_center_top_lip_x', 'mouth_center_top_lip_y'),
        ('mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y')
    ]

    # Obtiene la representación de la imagen y la convierte en un array de valores
    image_values = np.fromstring(row['Image'], sep=' ', dtype=int)

    # Realiza el reshape de la imagen a 96x96
    image_array = image_values.reshape(96, 96)

    # Muestra la imagen
    plt.imshow(image_array, cmap='gray')

    # Agrega puntos rojos en las coordenadas
    for x_col, y_col in points:
        x = row[x_col]
        y = row[y_col]
        plt.scatter(x, y, color='red', marker='o')

    # Ajusta el aspecto de la imagen
    plt.axis('off')

    # Muestra la imagen con los puntos rojos
    plt.show()
