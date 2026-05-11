import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eigenfaces import EigenfacesPCA
import matplotlib
matplotlib.use('TkAgg')

def main():
    print("=== Entrenamiento con PCA - Eigenfaces ===\n")

    # Inicializar PCA con 10 componentes
    pca = EigenfacesPCA(n_components=10)

    # Cargar imágenes de entrenamiento
    print("Cargando imágenes...")
    images, labels = pca.load_images_from_folder("data/training")

    if len(images) == 0:
        print("No se encontraron imágenes en data/training")
        return

    # Entrenar el modelo
    pca.train(images, labels)

    # Mostrar resultados matemáticos
    print("\n¿Qué deseas visualizar?")
    print("1 → Rostro promedio")
    print("2 → Eigenfaces")
    print("3 → Gráfica de eigenvalores")
    print("4 → Todo")

    opcion = input("\nElige una opción: ").strip()

    if opcion == "1":
        pca.show_mean_face()
    elif opcion == "2":
        pca.show_eigenfaces()
    elif opcion == "3":
        pca.show_eigenvalues_chart()
    elif opcion == "4":
        pca.show_mean_face()
        pca.show_eigenfaces()
        pca.show_eigenvalues_chart()
    else:
        print("Opción no válida")

if __name__ == "__main__":
    main()