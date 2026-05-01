# Sistema de Reconocimiento Facial

Sistema de reconocimiento facial en tiempo real desarrollado con Python.

## Requisitos

- Python 3.11
- Cámara web

## Instalación

1. Clona el repositorio:
   git clone https://github.com/tu-usuario/facial-recognition-system.git
   cd facial-recognition-system

2. Crea y activa el entorno virtual:
   python -m venv venv
   venv\Scripts\Activate.ps1

3. Instala las dependencias:
   pip install -r requirements.txt

## Cómo usar

Ejecuta el sistema:
   python src/main.py

### Opciones disponibles

- **Opción 1** — Registrar nuevo rostro: escribe el nombre, mira a la cámara y presiona ESPACIO para capturar
- **Opción 2** — Iniciar reconocimiento: abre la cámara y reconoce rostros en tiempo real
- **Q** — Salir del reconocimiento

## Estructura del proyecto

facial-recognition-system/
├── src/
│   ├── main.py        # Punto de entrada del sistema
│   ├── detector.py    # Lógica de detección y reconocimiento
│   └── utils.py       # Funciones de apoyo
├── data/              # Rostros registrados (no se sube a GitHub)
├── models/            # Modelos