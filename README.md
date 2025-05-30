# VisionGuardAI
# VisionGuardAI

VisionGuardAI es una aplicación backend diseñada para asistir a personas con discapacidad visual mediante tecnologías de visión artificial. Su función principal es recibir imágenes o frames en formato Base64, procesarlas con modelos avanzados como YOLO y devolver descripciones detalladas del entorno. Este backend no captura imágenes desde la cámara ni genera salidas de audio directamente, sino que se encarga del procesamiento y análisis de las imágenes recibidas.

## 🚀 Características
- 🔍 **Detección de objetos** en imágenes enviadas en formato Base64.
- 📝 **Generación de descripciones** detalladas del entorno basado en la detección de objetos.
- ⚡ **Procesamiento eficiente** utilizando modelos de inteligencia artificial optimizados.
- 🛠 **Arquitectura modular** con backend en Python.

## 📂 Estructura del Proyecto
```
VisionGuardAI/
│── app/
│   ├── config.py
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── utils/
│── yolov8n.pt
│── requirements.txt
│── README.md
```

## 🛠 Tecnologías Usadas
- Python (FastAPI)
- YOLO (You Only Look Once) para detección de objetos
- OpenCV para procesamiento de imágenes
- NumPy para manipulación de matrices de imágenes

## 📦 Instalación
1. Clona el repositorio:
   ```sh
   git clone https://github.com/Aldoveloper/VisionGuardAI.git
   ```
2. Ingresa al directorio del proyecto:
   ```sh
   cd VisionGuardAI
   ```
3. Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Uso
Para ejecutar la aplicación, simplemente usa el siguiente comando:
```sh
python app/main.py
```

## 📜 Licencia
Este proyecto está bajo la licencia [MIT](LICENSE).

## 📞 Contacto
Si tienes preguntas o sugerencias, contáctame en **florezaldo10@gmail.com**.

