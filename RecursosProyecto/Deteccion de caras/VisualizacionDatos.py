import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import random


def mostrar_imagenes_con_puntos(df, num_images=10, point_size=5):
    # Inicializa la figura para mostrar las imágenes
    fig, axs = plt.subplots(nrows=2, ncols=num_images, figsize=(15, 4))

    # Mostrar las primeras 10 imágenes sin puntos
    for i in range(num_images):
        # Obtiene la representación de la imagen y la convierte en un array de valores
        image_array = df.loc[i, 'Image'].reshape(96, 96)

        # Realiza el reshape de la imagen a 96x96
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
    # df_copy['Image'] = df_copy['Image'].apply(string_to_array)
    df_copy['Image'] = df_copy['Image'].apply(lambda x: x.reshape(96, 96))

    # Horizontal Flip - Damos la vuelta a las imágenes entorno al eje x (columnas)
    df_aumentado = df_copy.copy()
    df_aumentado['Image'] = df_aumentado['Image'].apply(lambda x: np.flip(x, axis=1).flatten())

    # Restar los valores iniciales de la coordenada x del ancho de la imagen (96)
    columns = df_aumentado.columns[:-1]  # Excluyendo la columna 'Image'
    for i in range(len(columns)):
        if i % 2 == 0:
            df_aumentado[columns[i]] = 96. - df_aumentado[columns[i]].astype(float)

    return df_aumentado


# Función para rotar imágenes y puntos
def rotar_imagenes_y_puntos(df, angle):
    # Copia del DataFrame original
    df_copy = df.copy()

    # Convierte la columna 'Image' en una matriz 2D y rota las imágenes
    df_copy['Image'] = df_copy['Image'].apply(lambda x: x.reshape(96, 96)).apply(
        lambda x: ndimage.rotate(x, -angle, reshape=False).flatten())

    # Rotar los puntos
    columns = df_copy.columns[:-1]  # Excluyendo la columna 'Image'
    for col_x, col_y in zip(columns[::2], columns[1::2]):
        for index, row in df_copy.iterrows():
            points = row[[col_x, col_y]]
            for i in range(0, len(points), 2):
                xy = np.array([points.iloc[i], points.iloc[i + 1]])
                xy_rot = rotar_puntos(xy, angle)
                points.iloc[i], points.iloc[i + 1] = xy_rot
            df_copy.at[index, col_x], df_copy.at[index, col_y] = points

    return df_copy


def aumentar_brillo(df):
    df_copy = df.copy()
    df_copy['Image'] = df_copy['Image'].apply(lambda x: np.clip(random.uniform(1.5, 2) * x, 0.0, 255.0))
    return df_copy


def decrementar_brillo(df):
    df_copy = df.copy()
    df_copy['Image'] = df_copy['Image'].apply(lambda x: np.clip(random.uniform(0, 1) * x, 0.0, 255.0))
    return df_copy


def borroso(df):
    df_copy = df.copy()
    df_copy['Image'] = df_copy['Image'].apply(lambda x: ndimage.gaussian_filter(x, sigma=1))
    return df_copy

def normalize_image(image):
    # Normaliza los valores de píxeles para que estén en el rango de 0 a 1
    return image / 255.0


# Leer el DataFrame original
df = pd.read_csv('../datasets_proyectos/puntos_faciales/data.csv')
df['Image'] = df['Image'].apply(string_to_array)

# Aumentar los datos
df_espejo = aumentar_datos(df)
df_resultado = pd.concat([df_espejo, df], ignore_index=True)

# Rotar imágenes y puntos
df_rotado_45 = rotar_imagenes_y_puntos(df_resultado, 45)
df_rotado_315 = rotar_imagenes_y_puntos(df_resultado, -45)

# Concatenar los DataFrames original y aumentado
array_df = [df_resultado, df_rotado_45, df_rotado_315]
df_resultado = pd.concat(array_df, ignore_index=True)

# Aumentar aleatoriamente el brillo de las imágenes
df_brillo_max = aumentar_brillo(df_resultado)

# Decrementar aleatoriamente el brillo de las imágenes
df_brillo_min = decrementar_brillo(df_resultado)

# Aplicar el filtro de desenfoque a las imágenes
df_borroso = borroso(df_resultado)

# Concatenar los DataFrames originales y aumentados
array_df = [df_resultado, df_brillo_max, df_brillo_min, df_borroso]
df_resultado = pd.concat(array_df, ignore_index=True)

# Mostrar las imágenes y los puntos
print(df_resultado.shape)
mostrar_imagenes_con_puntos(df_resultado, num_images=4, point_size=5)

# Aplicar normalización a los píxeles de la imagen
df_resultado['Image'] = df_resultado['Image'].apply(normalize_image)

# Guardar el DataFrame resultado en un archivo CSV
df_resultado.to_feather('resultado_aumentado.feather')
