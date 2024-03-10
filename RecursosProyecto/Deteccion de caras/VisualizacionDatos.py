import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mostrar_imagenes_con_puntos(df, num_images=10, point_size=5):
    # Inicializa la figura para mostrar las imágenes
    fig, axs = plt.subplots(nrows=2, ncols=num_images, figsize=(15, 4))

    # Mostrar las primeras 10 imágenes sin puntos
    for i in range(num_images):
        # Obtiene la representación de la imagen y la convierte en un array de valores
#        image_values = np.fromstring(df.loc[i, 'Image'], sep=' ', dtype=int)

        # Realiza el reshape de la imagen a 96x96
        image_array = df.loc[i, 'Image'].reshape(96, 96)

        # Ajusta el aspecto de la imagen
        axs[0, i].imshow(image_array, cmap='gray')
        axs[0, i].axis('off')

    # Mostrar las mismas 10 imágenes con puntos más pequeños
    for i in range(num_images):
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
        #image_values = np.fromstring(df.loc[i, 'Image'], sep=' ', dtype=int)

        # Realiza el reshape de la imagen a 96x96
        image_array = df.loc[i, 'Image'].reshape(96, 96)

        # Agrega puntos rojos en las coordenadas con puntos más pequeños
        for x_col, y_col in points:
            x = df.loc[i, x_col]
            y = df.loc[i, y_col]
            axs[1, i].scatter(x, y, color='red', marker='o', s=point_size)

        # Ajusta el aspecto de la imagen
        axs[1, i].imshow(image_array, cmap='gray')
        axs[1, i].axis('off')

    # Muestra las imágenes
    plt.tight_layout()
    plt.show()


def rotar_puntos(points, angle):
    # Cambia los puntos en el plano de manera que la rotación está justo en el origen
    # nuestra imagen es de 96*96 ,así que restamos 48
    points = points - 48

    # matriz de rotación
    # R = [ [cos(t), -sin(t)],[sin(t),cos(t)]
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # rotar los puntos
    for i in range(0, len(points), 2):
        xy = np.array([points[i], points[i + 1]])
        xy_rot = R @ xy
        points[i], points[i + 1] = xy_rot

    # volver al origen del centro de rotación
    points = points + 48
    return points

def string_to_array(image_string):
    return np.fromstring(image_string, dtype=int, sep=' ')

def aumentar_datos(df):
    # Copia del DataFrame original
    df_copy = df.copy()

    # Convierte la columna 'Image' en una matriz 2D
    df_copy['Image'] = df_copy['Image'].apply(string_to_array)
    df_copy['Image'] = df_copy['Image'].apply(lambda x: x.reshape(96, 96))

    # Horizontal Flip - Damos la vuelta a las imágenes entorno al eje x (columnas)
    df_aumentado = df_copy.copy()
    df_aumentado['Image'] = df_aumentado['Image'].apply(lambda x: np.flip(x, axis=1).flatten())

    # Restar los valores iniciales de la coordenada x del ancho de la imagen (96)
    columns = df_aumentado.columns[:-1]  # Excluyendo la columna 'Image'
    for i in range(len(columns)):
        if i % 2 == 0:
            df_aumentado[columns[i]] = 96. - df_aumentado[columns[i]].astype(float)

    # Concatenar los DataFrames original y aumentado
    df_resultado = pd.concat([df_copy, df_aumentado], ignore_index=True)

    return df_resultado


df = pd.read_csv('../datasets_proyectos/puntos_faciales/data.csv')
df_aumentado = aumentar_datos(df)
print(df_aumentado.shape, df.shape)
mostrar_imagenes_con_puntos(df_aumentado)
