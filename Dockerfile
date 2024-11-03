# Usa una versión compatible de Python con TensorFlow
FROM python:3.11.5

# Instala dependencias del sistema necesarias para OpenCV y otras bibliotecas
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configura el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY . /app

# Instala las dependencias de Python en requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Establece las variables de entorno para producción de Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expone el puerto para Flask
EXPOSE 5000

# Ejecuta la aplicación Flask en Waitress para producción
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]
