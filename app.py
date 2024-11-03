from flask import Flask, request, jsonify
from Scripts.model_api import ejecutar_modelo  # Asegúrate de que esta ruta es correcta
from PIL import Image
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si hay un archivo en la solicitud
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Verificar que el archivo es una imagen
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Leer la imagen en memoria y procesarla
        img = Image.open(file.stream)

        # Ejecutar el modelo
        predicted_labels, confidence_scores, processed_image = ejecutar_modelo(img)

        # Convertir todos los elementos de numpy.ndarray a listas
        predicted_labels = list(predicted_labels)  # Convertir las etiquetas a lista si es un array
        confidence_scores = [float(score) for score in confidence_scores]  # Asegurarse de que sea una lista de floats
        processed_image_list = processed_image.tolist()  # Convertir la imagen procesada a lista

        # Preparar la respuesta con datos serializables
        response = {
            "predicted_labels": predicted_labels,
            "confidence_scores": confidence_scores,
            "processed_image": processed_image_list  # Solo si es necesario; si no, exclúyelo
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
