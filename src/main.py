import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import FaceDetector
from utils import save_data, load_data, draw_results

def registrar_rostro(detector, video):
    name = input("\nEscribe el nombre de la persona: ").strip()
    if not name:
        return
    print(f"Mira a la cámara y presiona ESPACIO para capturar el rostro de {name}...")
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.putText(frame, "ESPACIO: capturar  |  ESC: cancelar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Registrar rostro", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Registro cancelado")
            break
        elif key == 32:  # ESPACIO
            temp_path = f"data/{name}_temp.jpg"
            os.makedirs("data", exist_ok=True)
            cv2.imwrite(temp_path, frame)
            detector.register_face(temp_path, name)
            save_data(detector.known_encodings, detector.known_names)
            os.remove(temp_path)
            break
    cv2.destroyAllWindows()

def main():
    detector = FaceDetector()
    detector.known_encodings, detector.known_names = load_data()

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("No se pudo abrir la cámara")
        return

    print("\n=== Sistema de Reconocimiento Facial ===")
    print("1 → Registrar nuevo rostro")
    print("2 → Iniciar reconocimiento")
    print("Q → Salir del reconocimiento")

    opcion = input("\nElige una opción (1 o 2): ").strip()

    if opcion == "1":
        registrar_rostro(detector, video)
        opcion2 = input("\n¿Iniciar reconocimiento ahora? (s/n): ").strip().lower()
        if opcion2 != "s":
            video.release()
            return

    print("\nIniciando reconocimiento... Presiona Q en la ventana para salir")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = detector.recognize_faces(frame)
        frame = draw_results(frame, results)

        cv2.putText(frame, "Q: Salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.imshow("Sistema de Reconocimiento Facial", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Cerrando sistema...")
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()