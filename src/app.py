import cv2
import sys
import os
import threading
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import FaceDetector
from utils import save_data, load_data, draw_results

# ─────────────────────────────────────────────
#  COLORES Y ESTILOS
# ─────────────────────────────────────────────
BG        = "#0d0f14"
PANEL     = "#13161e"
ACCENT    = "#00f0a0"
ACCENT2   = "#00b8ff"
DANGER    = "#ff4060"
TEXT      = "#e8eaf0"
SUBTEXT   = "#6b7080"
BORDER    = "#1e2230"

# ─────────────────────────────────────────────
#  APP PRINCIPAL
# ─────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Reconocimiento Facial")
        self.geometry("900x620")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.detector = FaceDetector()
        self.detector.known_encodings, self.detector.known_names = load_data()

        self.video      = None
        self.running    = False
        self.frame_count = 0

        # Contenedor de páginas
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
#  HELPERS DE WIDGETS
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

def make_label(parent, text, size=12, color=TEXT, bold=False):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, bg=BG, fg=color,
                    font=("Courier New", size, weight))

def card(parent, width, height):
    f = tk.Frame(parent, bg=PANEL, width=width, height=height,
                 highlightbackground=BORDER, highlightthickness=1)
    f.pack_propagate(False)
    return f


# ─────────────────────────────────────────────
#  PÁGINA 1 — MENÚ PRINCIPAL
# ─────────────────────────────────────────────
class MenuPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG)
        self.controller = controller

        # Título
        tk.Label(self, text="◈ FACE ID", bg=BG, fg=ACCENT,
                 font=("Courier New", 32, "bold")).pack(pady=(80, 4))
        tk.Label(self, text="Sistema de Reconocimiento Facial",
                 bg=BG, fg=SUBTEXT, font=("Courier New", 11)).pack(pady=(0, 60))

        # Botones
        make_btn(self, "  + Registrar Rostro",
                 lambda: controller.show("RegisterPage"), ACCENT).pack(pady=10)
        make_btn(self, "  ▶ Iniciar Reconocimiento",
                 lambda: controller.show("RecognizePage"), ACCENT2).pack(pady=10)

        # Footer
        tk.Label(self, text="presiona una opción para continuar",
                 bg=BG, fg=SUBTEXT, font=("Courier New", 9)).pack(side="bottom", pady=20)


# ─────────────────────────────────────────────
#  PÁGINA 2 — REGISTRAR ROSTRO
# ─────────────────────────────────────────────
class RegisterPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG)
        self.controller = controller
        self.capturing  = False

        # ── Layout: izquierda (cámara) + derecha (controles)
        left = tk.Frame(self, bg=BG)
        left.pack(side="left", padx=30, pady=30)

        right = tk.Frame(self, bg=BG)
        right.pack(side="left", padx=20, pady=30, fill="y")

        # Cámara
        tk.Label(left, text="CÁMARA EN VIVO", bg=BG, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(anchor="w", pady=(0, 8))
        self.cam_label = tk.Label(left, bg="#080a0f",
                                  width=640, height=400)
        self.cam_label.pack()

        hint = tk.Label(left, text="[ ESPACIO ] capturar  ·  [ ESC ] cancelar",
                        bg=BG, fg=SUBTEXT, font=("Courier New", 9))
        hint.pack(pady=6)

        # Controles derecha
        tk.Label(right, text="◈ REGISTRAR", bg=BG, fg=ACCENT,
                 font=("Courier New", 16, "bold")).pack(anchor="w", pady=(10, 30))

        tk.Label(right, text="Nombre:", bg=BG, fg=TEXT,
                 font=("Courier New", 11)).pack(anchor="w")
        self.name_var = tk.StringVar()
        entry = tk.Entry(right, textvariable=self.name_var,
                         bg=PANEL, fg=TEXT, insertbackground=ACCENT,
                         font=("Courier New", 12), relief="flat",
                         highlightbackground=BORDER, highlightthickness=1,
                         width=18)
        entry.pack(pady=(4, 20), ipady=8)

        self.status = tk.Label(right, text="", bg=BG, fg=ACCENT,
                               font=("Courier New", 10), wraplength=200)
        self.status.pack(pady=10)

        make_btn(right, "← Volver", self.volver, SUBTEXT, 160).pack(side="bottom", pady=10)

        # Bind teclado
        self.bind_all("<space>",  self.capturar)
        self.bind_all("<Escape>", self.volver_key)

    def on_show(self):
        self.name_var.set("")
        self.status.config(text="Escribe el nombre y\npresiona ESPACIO", fg=SUBTEXT)
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
            self.cam_label.config(image=imgtk, width=560, height=400)
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

        self.status.config(text="Procesando...", fg=ACCENT2)
        self.update()

        temp_path = f"data/{name}_temp.jpg"
        os.makedirs("data", exist_ok=True)
        cv2.imwrite(temp_path, self._frame)

        det = self.controller.detector
        det.register_face(temp_path, name)
        save_data(det.known_encodings, det.known_names)
        os.remove(temp_path)

        self.status.config(text=f"✓ {name} registrado\ncorrectamente", fg=ACCENT)

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
        self.controller   = controller
        self.recognizing  = False
        self.frame_count  = 0

        # Título
        tk.Label(self, text="◈ RECONOCIMIENTO EN VIVO", bg=BG, fg=ACCENT2,
                 font=("Courier New", 14, "bold")).pack(pady=(20, 4))
        tk.Label(self, text="el sistema identificará rostros automáticamente",
                 bg=BG, fg=SUBTEXT, font=("Courier New", 9)).pack()

        # Cámara
        self.cam_label = tk.Label(self, bg="#080a0f")
        self.cam_label.pack(pady=14)

        # Status
        self.status = tk.Label(self, text="Buscando rostros...", bg=BG, fg=SUBTEXT,
                               font=("Courier New", 11))
        self.status.pack(pady=6)

        # Botón volver
        make_btn(self, "← Volver al Menú", self.volver, SUBTEXT).pack(pady=10)

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

        # Reconocer solo 1 de cada 3 frames
        if self.frame_count % 3 == 0:
            results = self.controller.detector.recognize_faces(frame)
            frame   = draw_results(frame, results)
            self._last_results = results

            # Si reconoció a alguien conocido → ir a bienvenida
            known = [r for r in results if r["name"] != "Desconocido"]
            if known:
                name = known[0]["name"]
                self.recognizing = False
                self.controller.close_camera()
                self.controller.frames["WelcomePage"].set_name(name)
                self.controller.show("WelcomePage")
                return
        else:
            # Reusar últimos resultados para no perder los recuadros
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
#  PÁGINA 4 — BIENVENIDA (después de reconocer)
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