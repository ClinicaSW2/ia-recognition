import numpy as np
import os
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelEncoder
import joblib

# Evitar notación científica
np.set_printoptions(suppress=True)

# Cargar el modelo y el codificador de etiquetas
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
model_path = os.path.join(base_path, "Model", "modelo_entrenado.keras")
label_encoder_path = os.path.join(base_path, "Model", "label_encoder.joblib")

model = load_model(model_path, compile=False)
label_encoder = joblib.load(label_encoder_path)

# Función para aplicar ecualización de histograma a una imagen en color
def equalize_histogram_color(image):
    img_np = np.array(image)
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_output)

# Función para aplicar segmentación por K-means a una imagen en color
def segment_image_kmeans_color(image, k=4):
    image_np = np.array(image)
    image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2Lab)
    pixel_values = image_lab.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_lab.shape)

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_Lab2RGB)
    return Image.fromarray(segmented_image)

def ejecutar_modelo(image):
    # Aplicar preprocesamiento a la imagen
    equalized_image = equalize_histogram_color(image)
    segmented_image = segment_image_kmeans_color(equalized_image, k=4)

    # Redimensionar y preparar la imagen para el modelo
    processed_image = segmented_image.resize((224, 224))
    processed_image = img_to_array(processed_image) / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)

    # Predicción y scores de confianza
    prediction = model.predict(processed_image)
    sorted_indices = np.argsort(prediction[0])[::-1]
    first_index = sorted_indices[0]
    second_index = sorted_indices[1]

    confidence_scores = [
        format(prediction[0][first_index], '.4f'),
        format(prediction[0][second_index], '.4f')
    ]
    predicted_labels = label_encoder.inverse_transform([first_index, second_index])

    return predicted_labels, confidence_scores, processed_image
