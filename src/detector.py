import face_recognition
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []

    def register_face(self, image_path, name):
        """Registra un rostro conocido con su nombre"""
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            self.known_encodings.append(encodings[0])
            self.known_names.append(name)
            print(f"Rostro de {name} registrado correctamente")
        else:
            print(f"No se encontró rostro en la imagen de {name}")

    def recognize_faces(self, frame):
        """Detecta y reconoce rostros en un frame de video"""
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Desconocido"
            if self.known_encodings:
                matches = face_recognition.compare_faces(self.known_encodings, encoding)
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_match = np.argmin(distances)
                if matches[best_match]:
                    name = self.known_names[best_match]

            # Escalar coordenadas de vuelta al tamaño original
            top, right, bottom, left = [coord * 4 for coord in location]
            results.append({
                "name": name,
                "location": (top, right, bottom, left)
            })
        return results