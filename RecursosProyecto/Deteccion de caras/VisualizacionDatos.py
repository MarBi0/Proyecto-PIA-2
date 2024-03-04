import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reemplaza 'tu_archivo.csv' con la ruta o nombre real de tu archivo CSV
ruta_del_csv = '../datasets_proyectos/puntos_faciales/data.csv'

# Cargar el archivo CSV en un DataFrame
dataframe = pd.read_csv(ruta_del_csv)

# Imprimir las primeras filas del DataFrame
print(dataframe.head())

# visualizar las imagenes

# Hacer reshape con del array de imagenes a 96x96
imagenes = dataframe['Image'].apply(lambda x: np.fromstring(x, sep=' '))
imagenes = np.vstack(imagenes)
imagenes = imagenes.reshape(-1, 96, 96)  # Cambiar a 94x94

# Visualizar las imágenes
num_imagenes_a_mostrar = 10  # Puedes ajustar este número según tus necesidades

plt.figure(figsize=(10, 5))
for i in range(num_imagenes_a_mostrar):
    plt.subplot(1, num_imagenes_a_mostrar, i + 1)
    plt.imshow(imagenes[i], cmap='gray')
    plt.title(f'Imagen {i + 1}')
    plt.axis('off')

plt.show()