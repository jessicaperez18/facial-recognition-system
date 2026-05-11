import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class EigenfacesPCA:
    def __init__(self, n_components=10):
        """
        n_components: cuántas eigenfaces usar
        Más componentes = más precisión pero más costo computacional
        """
        self.n_components = n_components
        self.mean_face = None          # Rostro promedio
        self.eigenfaces = None         # Eigenvectores (las eigenfaces)
        self.eigenvalues = None        # Eigenvalores
        self.projections = None        # Rostros proyectados
        self.labels = []               # Nombres de cada rostro
        self.image_size = (100, 100)   # Tamaño estándar de imagen

    def load_images_from_folder(self, folder_path):
        """Carga todas las imágenes de una carpeta"""
        images = []
        labels = []

        for person_name in os.listdir(folder_path):
            person_path = os.path.join(folder_path, person_name)
            if not os.path.isdir(person_path):
                continue
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.image_size)
                    images.append(img)
                    labels.append(person_name)

        print(f"Imágenes cargadas: {len(images)}")
        return images, labels

    def train(self, images, labels):
        """Entrena el modelo PCA paso a paso"""
        self.labels = labels
        n_images = len(images)
        h, w = self.image_size

        # PASO 1: Convertir cada imagen a un vector de píxeles
        print("\n--- PASO 1: Aplanando imágenes a vectores ---")
        X = np.array([img.flatten() for img in images], dtype=np.float64)
        print(f"Matriz de datos X: {X.shape} ({n_images} imágenes x {h*w} píxeles)")

        # PASO 2: Calcular el rostro promedio
        print("\n--- PASO 2: Calculando rostro promedio ---")
        self.mean_face = np.mean(X, axis=0)
        print(f"Rostro promedio calculado (vector de {len(self.mean_face)} valores)")

        # PASO 3: Centralizar los datos (restar el promedio)
        print("\n--- PASO 3: Centralizando datos ---")
        X_centered = X - self.mean_face
        print(f"Datos centrados: {X_centered.shape}")

        # PASO 4: Calcular la matriz de covarianza
        print("\n--- PASO 4: Calculando matriz de covarianza ---")
        # Truco matemático: usar X·Xᵀ en lugar de Xᵀ·X (más eficiente)
        cov_matrix = np.dot(X_centered, X_centered.T) / n_images
        print(f"Matriz de covarianza: {cov_matrix.shape}")

        # PASO 5: Calcular eigenvalores y eigenvectores
        print("\n--- PASO 5: Calculando eigenvalores y eigenvectores ---")
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Ordenar de mayor a menor eigenvalor
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        print(f"Eigenvalores obtenidos: {len(eigenvalues)}")
        print(f"\nTop {self.n_components} eigenvalores:")
        for i, val in enumerate(eigenvalues[:self.n_components]):
            varianza = (val / np.sum(eigenvalues)) * 100
            print(f"  λ{i+1} = {val:.2f}  →  explica el {varianza:.1f}% de la varianza")

        # PASO 6: Convertir eigenvectores al espacio original
        print("\n--- PASO 6: Calculando Eigenfaces ---")
        eigenfaces = np.dot(X_centered.T, eigenvectors)

        # Normalizar cada eigenface
        for i in range(eigenfaces.shape[1]):
            eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])

        # Guardar solo los n_components más importantes
        self.eigenvalues = eigenvalues[:self.n_components]
        self.eigenfaces = eigenfaces[:, :self.n_components]

        # PASO 7: Proyectar todos los rostros en el espacio de eigenfaces
        print("\n--- PASO 7: Proyectando rostros ---")
        self.projections = np.dot(X_centered, self.eigenfaces)
        print(f"Proyecciones calculadas: {self.projections.shape}")
        print("\n✅ Entrenamiento completado")

    def recognize(self, image):
        """Reconoce un rostro usando distancia euclidiana"""
        img = cv2.resize(image, self.image_size)
        img_vector = img.flatten().astype(np.float64)

        # Centralizar y proyectar
        img_centered = img_vector - self.mean_face
        projection = np.dot(img_centered, self.eigenfaces)

        # Calcular distancia con cada rostro conocido
        distances = np.linalg.norm(self.projections - projection, axis=1)
        best_match = np.argmin(distances)
        min_distance = distances[best_match]

        threshold = 3000
        if min_distance < threshold:
            return self.labels[best_match], min_distance
        return "Desconocido", min_distance

    def show_mean_face(self):
        """Muestra el rostro promedio"""
        mean_img = self.mean_face.reshape(self.image_size)
        mean_img = cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX)
        plt.figure(figsize=(4, 4))
        plt.imshow(mean_img.astype(np.uint8), cmap="gray")
        plt.title("Rostro Promedio (Mean Face)")
        plt.axis("off")
        plt.show()

    def show_eigenfaces(self):
        """Muestra las eigenfaces más importantes"""
        n = min(self.n_components, 10)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle("Eigenfaces (Eigenvectores principales)", fontsize=14)

        for i, ax in enumerate(axes.flat):
            if i < n:
                eigenface = self.eigenfaces[:, i].reshape(self.image_size)
                eigenface = cv2.normalize(eigenface, None, 0, 255, cv2.NORM_MINMAX)
                ax.imshow(eigenface.astype(np.uint8), cmap="gray")
                varianza = (self.eigenvalues[i] / np.sum(self.eigenvalues)) * 100
                ax.set_title(f"λ{i+1} = {self.eigenvalues[i]:.1f}\n{varianza:.1f}% varianza",
                           fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def show_eigenvalues_chart(self):
        """Muestra gráfica de eigenvalores"""
        varianzas = (self.eigenvalues / np.sum(self.eigenvalues)) * 100
        varianza_acumulada = np.cumsum(varianzas)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.bar(range(1, len(varianzas)+1), varianzas, color="steelblue")
        ax1.set_title("Varianza explicada por cada Eigenvalor")
        ax1.set_xlabel("Componente principal")
        ax1.set_ylabel("% Varianza")

        ax2.plot(range(1, len(varianza_acumulada)+1), varianza_acumulada,
                marker="o", color="coral")
        ax2.axhline(y=95, color="green", linestyle="--", label="95% varianza")
        ax2.set_title("Varianza acumulada")
        ax2.set_xlabel("Número de componentes")
        ax2.set_ylabel("% Varianza acumulada")
        ax2.legend()

        plt.tight_layout()
        plt.show()