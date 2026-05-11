import cv2
import os

def capturar_fotos(nombre, n_fotos=10):
    """Captura fotos desde la cámara para entrenar el modelo"""
    
    carpeta = f"data/training/{nombre}"
    os.makedirs(carpeta, exist_ok=True)

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("No se pudo abrir la cámara")
        return

    print(f"\nCapturando {n_fotos} fotos para {nombre}")
    print("Presiona ESPACIO para capturar cada foto")
    print("Presiona ESC para cancelar\n")

    contador = 0

    while contador < n_fotos:
        ret, frame = video.read()
        if not ret:
            break

        cv2.putText(frame, f"Foto {contador+1}/{n_fotos} — ESPACIO: capturar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Capturar fotos", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Cancelado")
            break
        elif key == 32:  # ESPACIO
            foto_path = f"{carpeta}/foto_{contador+1}.jpg"
            cv2.imwrite(foto_path, frame)
            print(f"✅ Foto {contador+1} guardada")
            contador += 1

    video.release()
    cv2.destroyAllWindows()
    print(f"\nTotal fotos capturadas: {contador}")

if __name__ == "__main__":
    nombre = input("Nombre de la persona: ").strip()
    n_fotos = int(input("Cuántas fotos quieres capturar (recomendado 10): ").strip())
    capturar_fotos(nombre, n_fotos)