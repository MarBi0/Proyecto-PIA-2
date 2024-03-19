from tensorflow import keras
import cv2 as cv

# Cargar el modelo
model = keras.models.load_model('modelo_puntos_faciales.h5')

face_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')

# abrir la cámara
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectar caras
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Recortar la cara
        face = gray[y:y + h, x:x + w]

        # Cambiar el tamaño de la imagen a 96x96
        face_resized = cv.resize(face, (96, 96))

        # Normalizar la imagen
        face_normalized = face_resized / 255.0

        # Convertir la imagen a un tensor 3D
        face_tensor = face_normalized.reshape(1, 96, 96, 1)

        # Predecir los puntos faciales
        points = model.predict(face_tensor)

        # Dibujar los puntos faciales
        for i in range(0, len(points[0]), 2):
            x_pred = int(points[0][i] * w / 96) + x
            y_pred = int(points[0][i + 1] * h / 96) + y
            cv.circle(frame, (x_pred, y_pred), 2, (0, 255, 0), -1)

    # Mostrar la imagen
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
