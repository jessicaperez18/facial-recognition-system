import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eigenfaces import EigenfacesPCA

class ReconocedorPCA:
    def __init__(self):
        self.pca = EigenfacesPCA(n_components=10)
        self.entrenado = False

    def entrenar(self):
        """Entrena el modelo PCA con las fotos de data/training"""
        images, labels = self.pca.load_images_from_folder("data/training")
        if len(images) == 0:
            print("No hay imágenes de entrenamiento")
            return False
        self.pca.train(images, labels)
        self.entrenado = True
        return True

    def reconocer_frame(self, frame):
        """Reconoce rostros en un frame usando PCA"""
        if not self.entrenado:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        results = []
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            name, distance = self.pca.recognize(face_img)
            results.append({
                "name": name,
                "location": (y, x+w, y+h, x),
                "distance": round(distance, 1)
            })
        return results