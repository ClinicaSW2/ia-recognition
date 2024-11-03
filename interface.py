import sys
import requests
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class DragDropWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.file_path = None  # Variable para almacenar la ruta de la imagen cargada

        # Etiqueta de arrastrar y soltar
        self.label = QLabel("Arrastra una imagen aquí")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedSize(300, 300)
        self.label.setStyleSheet("border: 2px dashed #4CAF50; background-color: #EBF7EB; font-size: 16px; color: black;")
        self.layout.addWidget(self.label)

        # Botón para abrir imagen desde archivo
        self.button = QPushButton("Selecciona una imagen")
        self.button.clicked.connect(self.open_file)
        self.layout.addWidget(self.button)

        # Botón para ejecutar el modelo
        self.run_button = QPushButton("Ejecutar Modelo")
        self.run_button.clicked.connect(self.on_new_button_clicked)
        self.layout.addWidget(self.run_button)

        self.setAcceptDrops(True)  # Permitir arrastrar y soltar

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.label.setStyleSheet("border: 2px solid #4CAF50; background-color: white;")

    def dragLeaveEvent(self, event):
        self.label.setStyleSheet("border: 2px dashed #4CAF50; background-color: #EBF7EB;")

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            self.file_path = url.toLocalFile()
            self.show_image(self.file_path)
            break
        self.label.setStyleSheet("border: 2px dashed #4CAF50; background-color: #EBF7EB;")

    def show_image(self, filepath):
        pixmap = QPixmap(filepath)
        pixmap = pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos de imagen (*.jpg *.jpeg *.png)")
        if file_path:
            self.file_path = file_path
            self.show_image(file_path)

    def on_new_button_clicked(self):
        # Verificar si hay una imagen cargada
        if not self.file_path:
            QMessageBox.warning(self, "Advertencia", "Por favor, carga una imagen primero.")
            return

        try:
            # Abrir la imagen como archivo para enviarla a la API
            with open(self.file_path, 'rb') as img_file:
                response = requests.post("http://127.0.0.1:5000/predict", files={"file": img_file})

            if response.status_code == 200:
                data = response.json()
                predicted_labels = data["predicted_labels"]
                confidence_scores = data["confidence_scores"]

                # Mostrar los resultados en la interfaz
                self.mostrar_resultados(predicted_labels, confidence_scores)
            else:
                QMessageBox.critical(self, "Error", f"Error en la API: {response.json().get('error')}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al conectar con la API: {str(e)}")

    def mostrar_resultados(self, predicted_labels, confidence_scores):
        # Crear un mensaje con los resultados
        result_text = "Resultados:\n"
        for label, confidence in zip(predicted_labels, confidence_scores):
            result_text += f"{label}: {confidence:.2f}\n"

        # Mostrar los resultados en un cuadro de mensaje
        QMessageBox.information(self, "Resultados del Modelo", result_text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocimiento de Enfermedades Oculares")
        self.resize(400, 400)

        # Configurar el widget central
        self.drag_drop_widget = DragDropWidget()
        self.setCentralWidget(self.drag_drop_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
