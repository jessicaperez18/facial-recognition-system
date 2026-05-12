from reconocedor_pca import ReconocedorPCA
from utils import save_data, load_data, draw_results
from detector import FaceDetector
import cv2
import sys
import os
import tkinter as tk
from PIL import Image, ImageTk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
#  COLORES Y ESTILOS
# ─────────────────────────────────────────────
BG = "#0d0f14"
PANEL = "#13161e"
ACCENT = "#00f0a0"
ACCENT2 = "#00b8ff"
ACCENT3 = "#ff9f00"
DANGER = "#ff4060"
TEXT = "#e8eaf0"
SUBTEXT = "#6b7080"
BORDER = "#1e2230"

# ─────────────────────────────────────────────
#  APP PRINCIPAL
# ─────────────────────────────────────────────


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Reconocimiento Facial")
        self.geometry("900x640")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.detector = FaceDetector()
        self.detector.known_encodings, self.detector.known_names = load_data()

        self.pca_detector = ReconocedorPCA()

        self.video = None
        self.running = False

        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for Page in (MenuPage, RegisterPage, RecognizePage, WelcomePage):
            frame = Page(container, self)
            self.frames[Page.__name__] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show("MenuPage")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def show(self, name):
        self.frames[name].tkraise()
        if hasattr(self.frames[name], "on_show"):
            self.frames[name].on_show()

    def open_camera(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True

    def close_camera(self):
        self.running = False
        if self.video:
            self.video.release()
            self.video = None

    def on_close(self):
        self.close_camera()
        self.destroy()


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def make_btn(parent, text, command, color=ACCENT, width=220):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=color, fg=BG, activebackground=color,
        font=("Courier New", 11, "bold"),
        relief="flat", cursor="hand2",
        width=width // 10, pady=10
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=_lighten(color)))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn


def _lighten(hex_color):
    r = min(255, int(hex_color[1:3], 16) + 30)
    g = min(255, int(hex_color[3:5], 16) + 30)
    b = min(255, int(hex_color[5:7], 16) + 30)
    return f"#{r:02x}{g:02x}{b:02x}"


# ─────────────────────────────────────────────
#  PÁGINA 1 — MENÚ PRINCIPAL
# ─────────────────────────────────────────────
class MenuPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG)
        self.controller = controller

        tk.Label(self, text="◈ FACE ID", bg=BG, fg=ACCENT,
                 font=("Courier New", 32, "bold")).pack(pady=(60, 4))
        tk.Label(self, text="Sistema de Reconocimiento Facial",
                 bg=BG, fg=SUBTEXT, font=("Courier New", 11)).pack(pady=(0, 40))

        # Método de reconocimiento
        tk.Label(self, text="MÉTODO DE RECONOCIMIENTO",
                 bg=BG, fg=SUBTEXT, font=("Courier New", 9)).pack(pady=(0, 10))

        self.metodo = tk.StringVar(value="face_recognition")
        frame_metodos = tk.Frame(self, bg=BG)
        frame_metodos.pack(pady=(0, 30))

        tk.Radiobutton(frame_metodos, text="face_recognition",
                       variable=self.metodo, value="face_recognition",
                       bg=BG, fg=ACCENT, selectcolor=PANEL,
                       activebackground=BG, activeforeground=ACCENT,
                       font=("Courier New", 11)).pack(side="left", padx=20)

        tk.Radiobutton(frame_metodos, text="PCA Eigenfaces",
                       variable=self.metodo, value="pca",
                       bg=BG, fg=ACCENT3, selectcolor=PANEL,
                       activebackground=BG, activeforeground=ACCENT3,
                       font=("Courier New", 11)).pack(side="left", padx=20)

        # Botones
        make_btn(self, "  + Registrar Rostro",
                 lambda: controller.show("RegisterPage"), ACCENT).pack(pady=8)
        make_btn(self, "  ▶ Iniciar Reconocimiento",
                 self.iniciar_reconocimiento, ACCENT2).pack(pady=8)

        tk.Label(self, text="presiona una opción para continuar",
                 bg=BG, fg=SUBTEXT, font=("Courier New", 9)).pack(side="bottom", pady=20)

    def iniciar_reconocimiento(self):
        metodo = self.metodo.get()
        self.controller.frames["RecognizePage"].set_metodo(metodo)

        if metodo == "pca":
            self.controller.frames["RecognizePage"].status.config(
                text="Entrenando PCA...", fg=ACCENT3)
            self.update()
            ok = self.controller.pca_detector.entrenar()
            if not ok:
                self.controller.frames["RecognizePage"].status.config(
                    text="⚠ No hay fotos de entrenamiento", fg=DANGER)
                return

        self.controller.show("RecognizePage")


# ─────────────────────────────────────────────
#  PÁGINA 2 — REGISTRAR ROSTRO
# ─────────────────────────────────────────────
class RegisterPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG)
        self.controller = controller
        self.capturing = False

        left = tk.Frame(self, bg=BG)
        left.pack(side="left", padx=30, pady=30)

        right = tk.Frame(self, bg=BG)
        right.pack(side="left", padx=20, pady=30, fill="y")

        tk.Label(left, text="CÁMARA EN VIVO", bg=BG, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(anchor="w", pady=(0, 8))
        self.cam_label = tk.Label(left, bg="#080a0f", width=560, height=400)
        self.cam_label.pack()

        tk.Label(left, text="[ ESPACIO ] capturar  ·  [ ESC ] cancelar",
                 bg=BG, fg=SUBTEXT, font=("Courier New", 9)).pack(pady=6)

        tk.Label(right, text="◈ REGISTRAR", bg=BG, fg=ACCENT,
                 font=("Courier New", 16, "bold")).pack(anchor="w", pady=(10, 30))

        tk.Label(right, text="Nombre:", bg=BG, fg=TEXT,
                 font=("Courier New", 11)).pack(anchor="w")
        self.name_var = tk.StringVar()
        tk.Entry(right, textvariable=self.name_var,
                 bg=PANEL, fg=TEXT, insertbackground=ACCENT,
                 font=("Courier New", 12), relief="flat",
                 highlightbackground=BORDER, highlightthickness=1,
                 width=18).pack(pady=(4, 20), ipady=8)

        self.status = tk.Label(right, text="", bg=BG, fg=ACCENT,
                               font=("Courier New", 10), wraplength=200)
        self.status.pack(pady=10)

        make_btn(right, "← Volver", self.volver, SUBTEXT,
                 160).pack(side="bottom", pady=10)

        self.bind_all("<space>",  self.capturar)
        self.bind_all("<Escape>", self.volver_key)

    def on_show(self):
        self.name_var.set("")
        self.status.config(
            text="Escribe el nombre y\npresiona ESPACIO", fg=SUBTEXT)
        self.controller.open_camera()
        self.capturing = True
        self._update_frame()

    def _update_frame(self):
        if not self.capturing or not self.controller.running:
            return
        ret, frame = self.controller.video.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((560, 400))
            imgtk = ImageTk.PhotoImage(img)
            self.cam_label.config(image=imgtk)
            self.cam_label.image = imgtk
            self._frame = frame
        self.after(15, self._update_frame)

    def capturar(self, event=None):
        name = self.name_var.get().strip()
        if not name:
            self.status.config(text="⚠ Escribe un nombre primero", fg=DANGER)
            return
        if not hasattr(self, "_frame"):
            self.status.config(text="⚠ Cámara no disponible", fg=DANGER)
            return

    # Inicializar contador de fotos si no existe
        if not hasattr(self, "_foto_count"):
            self._foto_count = 0
            self._foto_name  = name

    # Si cambiaron el nombre reiniciar contador
        if self._foto_name != name:
            self._foto_count = 0
            self._foto_name  = name

        total_fotos = 5
        self._foto_count += 1

        self.status.config(
            text=f"Capturando foto {self._foto_count}/{total_fotos}...",
            fg=ACCENT2)
        self.update()

        training_path = f"data/training/{name}"
        os.makedirs(training_path, exist_ok=True)
        foto_path = f"{training_path}/foto_{self._foto_count}.jpg"
        cv2.imwrite(foto_path, self._frame)

        if self._foto_count >= total_fotos:
            temp_path = f"data/{name}_temp.jpg"
            cv2.imwrite(temp_path, self._frame)
            det = self.controller.detector
            det.register_face(temp_path, name)
            save_data(det.known_encodings, det.known_names)
            os.remove(temp_path)
            self.controller.pca_detector = ReconocedorPCA()
            self.controller.pca_detector.entrenar()
            self.status.config(
                text=f"✓ {name} registrado\n5 fotos guardadas", fg=ACCENT)
            self._foto_count = 0

        else:
            self.status.config(
                text=f"Foto {self._foto_count}/{total_fotos} ✓\nPresiona ESPACIO para continuar",
                fg=ACCENT2)
    def volver(self):
            self.capturing = False
            self.controller.close_camera()
            self.unbind_all("<space>")
            self.unbind_all("<Escape>")
            self.controller.show("MenuPage")

    def volver_key(self, event=None):
            self.volver()    


# ─────────────────────────────────────────────
#  PÁGINA 3 — RECONOCIMIENTO EN VIVO
# ─────────────────────────────────────────────
class RecognizePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG)
        self.controller = controller
        self.recognizing = False
        self.metodo = "face_recognition"
        self.frame_count = 0

        tk.Label(self, text="◈ RECONOCIMIENTO EN VIVO", bg=BG, fg=ACCENT2,
                 font=("Courier New", 14, "bold")).pack(pady=(20, 4))

        self.metodo_label = tk.Label(self, text="", bg=BG, fg=SUBTEXT,
                                     font=("Courier New", 9))
        self.metodo_label.pack()

        self.cam_label = tk.Label(self, bg="#080a0f")
        self.cam_label.pack(pady=14)

        self.status = tk.Label(self, text="Buscando rostros...", bg=BG, fg=SUBTEXT,
                               font=("Courier New", 11))
        self.status.pack(pady=6)

        make_btn(self, "← Volver al Menú", self.volver, SUBTEXT).pack(pady=10)

    def set_metodo(self, metodo):
        self.metodo = metodo
        if metodo == "pca":
            self.metodo_label.config(text="método: PCA Eigenfaces", fg=ACCENT3)
        else:
            self.metodo_label.config(
                text="método: face_recognition", fg=ACCENT)

    def on_show(self):
        self.recognizing = True
        self.frame_count = 0
        self.controller.open_camera()
        self._update_frame()

    def _update_frame(self):
        if not self.recognizing or not self.controller.running:
            return

        ret, frame = self.controller.video.read()
        if not ret:
            self.after(15, self._update_frame)
            return

        self.frame_count += 1

        if self.frame_count % 3 == 0:
            if self.metodo == "pca":
                results = self.controller.pca_detector.reconocer_frame(frame)
            else:
                results = self.controller.detector.recognize_faces(frame)

            frame = draw_results(frame, results)
            self._last_results = results

            known = [r for r in results if r["name"] != "Desconocido"]
            if known:
                name = known[0]["name"]
                self.recognizing = False
                self.controller.close_camera()
                self.controller.frames["WelcomePage"].set_name(name)
                self.controller.show("WelcomePage")
                return
        else:
            if hasattr(self, "_last_results"):
                frame = draw_results(frame, self._last_results)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((640, 420))
        imgtk = ImageTk.PhotoImage(img)
        self.cam_label.config(image=imgtk, width=640, height=420)
        self.cam_label.image = imgtk

        self.after(15, self._update_frame)

    def volver(self):
        self.recognizing = False
        self.controller.close_camera()
        self.controller.show("MenuPage")


# ─────────────────────────────────────────────
#  PÁGINA 4 — BIENVENIDA
# ─────────────────────────────────────────────
class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG)
        self.controller = controller

        tk.Label(self, text="✓", bg=BG, fg=ACCENT,
                 font=("Courier New", 64)).pack(pady=(80, 10))
        tk.Label(self, text="IDENTIDAD VERIFICADA", bg=BG, fg=ACCENT,
                 font=("Courier New", 18, "bold")).pack()

        self.name_label = tk.Label(self, text="", bg=BG, fg=TEXT,
                                   font=("Courier New", 28, "bold"))
        self.name_label.pack(pady=20)

        tk.Label(self, text="Bienvenido al sistema", bg=BG, fg=SUBTEXT,
                 font=("Courier New", 11)).pack()

        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=50)

        make_btn(btn_frame, "↺ Reconocer otro",
                 lambda: controller.show("RecognizePage"), ACCENT2).pack(side="left", padx=10)
        make_btn(btn_frame, "⌂ Menú principal",
                 lambda: controller.show("MenuPage"), ACCENT).pack(side="left", padx=10)

    def set_name(self, name):
        self.name_label.config(text=name.upper())


# ─────────────────────────────────────────────
#  ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
