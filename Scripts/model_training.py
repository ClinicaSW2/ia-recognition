import os
import numpy as np
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Función para cargar imágenes
def load_images_from_folder(folder, target_size):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))

    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(class_name)

    return np.array(images), np.array(labels)

# Preparación de los datos
training_path = os.path.join('Training')
X_data_1, Y_data = load_images_from_folder(training_path, (224, 224))
X_data = X_data_1 / 255.0
label_encoder = LabelEncoder()
Y_data_encoded = label_encoder.fit_transform(Y_data)
num_classes = len(label_encoder.classes_)

# Creación del directorio 'Model' y guardado del codificador de etiquetas
model_dir = 'Model'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(label_encoder, os.path.join('Model', 'label_encoder.joblib'))

# División de los datos de entrenamiento y validación
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data_encoded, test_size=0.25, random_state=1234)

# Definición y compilación del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Configuración de EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenamiento del modelo
model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_data=(X_val, Y_val), callbacks=[early_stopping])

# Guardado del modelo en el formato moderno de Keras
model.save(os.path.join('Model', 'modelo_entrenado.keras'))
