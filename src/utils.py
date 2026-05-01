import pickle
import os

def save_data(encodings, names, filepath="data/faces_data.pkl"):
    """Guarda los rostros registrados en un archivo"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print("Datos guardados correctamente")

def load_data(filepath="data/faces_data.pkl"):
    """Carga los rostros registrados desde un archivo"""
    if not os.path.exists(filepath):
        print("No hay datos guardados aún")
        return [], []
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def draw_results(frame, results):
    """Dibuja los recuadros y nombres en el frame"""
    import cv2
    for result in results:
        name = result["name"]
        top, right, bottom, left = result["location"]

        # Color verde para conocidos, rojo para desconocidos
        color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame