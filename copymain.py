from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from collections import Counter
import re
import uvicorn
import numpy as np
import cv2
import base64
from ultralytics import YOLO
import asyncio
import logging
import re
import pytesseract
import easyocr


# Diccionario de traducción de nombres de objetos
OBJETO_NOMBRES_ES = {
    "person": "persona",
    "car": "carro",
    "bicycle": "bicicleta",
    "motorcycle": "motocicleta",
    "bus": "autobús",
    "truck": "camión",
    "traffic light": "semáforo",
    "stop sign": "señal de alto",
    "dog": "perro",
    "cat": "gato",
    "backpack": "mochila",
    "umbrella": "paraguas",
    "handbag": "bolso",
    "suitcase": "maleta",
    "bottle": "botella",
    "wine glass": "copa de vino",
    "cup": "taza",
    "fork": "tenedor",
    "knife": "cuchillo",
    "spoon": "cuchara",
    "bowl": "tazón",
    "banana": "plátano",
    "apple": "manzana",
    "sandwich": "sándwich",
    "orange": "naranja",
    "broccoli": "brócoli",
    "carrot": "zanahoria",
    "hot dog": "perro caliente",
    "pizza": "pizza",
    "donut": "rosquilla",
    "cake": "pastel",
    "chair": "silla",
    "sofa": "sofá",
    "pottedplant": "planta en maceta",
    "bed": "cama",
    "diningtable": "mesa de comedor",
    "toilet": "inodoro",
    "tvmonitor": "monitor de TV",
    "laptop": "portátil",
    "mouse": "ratón",
    "remote": "control remoto",
    "keyboard": "teclado",
    "cell phone": "teléfono celular",
    "microwave": "microondas",
    "oven": "horno",
    "toaster": "tostadora",
    "sink": "fregadero",
    "refriger":"refrigerador",
    "book": "libro",
    "clock": "reloj",
    "vase": "jarrón",
    "scissors": "tijeras",

}

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el modelo YOLOv8cls
MODEL_PATH = "yolov8n.pt"
if not MODEL_PATH:
    raise FileNotFoundError("El modelo YOLO no se encontró.")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error al cargar YOLO: {e}")

# Iniciar la aplicación FastAPI
app = FastAPI()

def clean_base64(data: str) -> str:
    """
    Si la cadena Base64 incluye el prefijo 'data:image/jpeg;base64,'
    (u otro similar), lo elimina.
    """
    return re.sub(r"^data:.*;base64,", "", data)


def detect_objects(image_bytes: bytes) -> dict:
    """
    Recibe bytes de imagen, los decodifica y usa YOLO para detectar objetos y su ubicación.
    Retorna un diccionario con los objetos detectados y sus posiciones.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("La imagen no se pudo decodificar.")
            return {"error": "No se pudo decodificar la imagen"}

        # Obtener dimensiones de la imagen
        height, width, _ = frame.shape

        # Realizar detección con YOLO
        results = model(frame)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                try:
                    cls = int(box.cls[0])  # Clase del objeto
                    label = model.names.get(cls, "desconocido")

                    # Obtener coordenadas del cuadro delimitador
                    x_center, y_center, w, h = box.xywh[0]  # (x, y, width, height)

                    # Determinar ubicación en la imagen
                    if x_center < width / 3:
                        position = "izquierda"
                    elif x_center > 2 * width / 3:
                        position = "derecha"
                    else:
                        position = "centro"

                    detected_objects.append({"label": label, "position": position})

                except Exception as e:
                    logger.error(f"Error procesando una caja: {e}")

        return {"detected_objects": detected_objects}

    except Exception as e:
        logger.error(f"Error en detect_objects: {e}")
        return {"error": "Error en detección"}



def detect_text_noise(detected_text: str) -> str:
    """
    Detecta si el texto extraído tiene demasiado ruido y devuelve una descripción más clara.
    """
    if not detected_text.strip():
        return ""  # No hay texto

    # Contar caracteres especiales y números
    special_chars = len(re.findall(r"[^\w\s]", detected_text))  # Caracteres raros
    numbers = len(re.findall(r"\d", detected_text))  # Números
    words = detected_text.split()

    # Si más del 40% del texto son caracteres especiales/números, es ruido
    if (special_chars + numbers) / len(detected_text) > 0.4 or len(words) < 3:
        return "Hay algunos letreros, pero el texto no es completamente legible."

    return detected_text


def format_objects(obj_list):
    """
    Formatea una lista de objetos detectados con artículos adecuados y corrección de gramática.
    """
    obj_count = Counter(obj_list)
    formatted_list = []

    for obj, count in obj_count.items():
        obj_es = OBJETO_NOMBRES_ES.get(obj, obj)  # Traducción a español

        # Determinar si es plural
        obj_plural = obj_es + "s" if obj_es.endswith(("a", "e", "o")) else obj_es
        article = "una" if obj_es.endswith("a") else "un"  # "una persona", "un carro"

        # Si es "persona", agregamos "a" para que suene más natural
        if obj_es == "persona":
            if count == 1:
                formatted_list.append(f"a {article} {obj_es}")
            else:
                formatted_list.append(f"a {count} {obj_plural}")
        else:
            if count == 1:
                formatted_list.append(f"{article} {obj_es}")
            else:
                formatted_list.append(f"{count} {obj_plural}")

    # Conectores adecuados
    if len(formatted_list) > 1:
        return ", ".join(formatted_list[:-1]) + " y " + formatted_list[-1]
    return formatted_list[0] if formatted_list else ""

def describe_scene(detected_objects: list, detected_text: str = "") -> str:
    """
    Genera una descripción con gramática mejorada y detecta texto malformado.
    """
    if not detected_objects and not detected_text:
        return "No se detectaron objetos ni texto en la imagen."

    description = ""

    # Agrupar objetos por ubicación y traducir nombres
    locations = {"izquierda": [], "centro": [], "derecha": []}

    for obj in detected_objects:
        label_es = OBJETO_NOMBRES_ES.get(obj["label"], obj["label"])  # Traducir nombres
        locations[obj["position"]].append(label_es)

    # Construcción de la descripción con mejor estructura
    if locations["izquierda"]:
        description += f"A tu izquierda veo {format_objects(locations['izquierda'])}. "
    if locations["centro"]:
        description += f"Frente a ti {format_objects(locations['centro'])}. "
    if locations["derecha"]:
        description += f"A tu derecha {format_objects(locations['derecha'])}. "

    # Advertencias específicas
    obstacles = {"carro", "persona", "bicicleta", "motocicleta", "camión", "autobús"}
    obstacles_detected = [obj for obj in detected_objects if obj["label"] in obstacles]

    if obstacles_detected:
        description += "Ten cuidado, hay obstáculos en tu camino. "

    # Mejorar la detección de texto
    corrected_text = detect_text_noise(detected_text)
    if corrected_text:
        description += f"También veo un letrero o texto que dice: '{corrected_text}'. "

    return description.strip()

reader = easyocr.Reader(['es', 'en'])  # Soporta español e inglés

def extract_text_from_image(image_bytes: bytes, use_easyocr=True) -> str:
    """
    Extrae texto de una imagen usando Tesseract o EasyOCR.
    
    :param image_bytes: Bytes de la imagen.
    :param use_easyocr: Si es True, usa EasyOCR; si es False, usa Tesseract.
    :return: Texto detectado en la imagen.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # Convertir a escala de grises

        if img is None:
            return "No se pudo procesar la imagen."

        if use_easyocr:
            result = reader.readtext(img, detail=0)  # Obtener solo el texto
            return " ".join(result)
        else:
            return pytesseract.image_to_string(img, lang='spa+eng')  # Extraer texto con Tesseract

    except Exception as e:
        return f"Error en OCR: {e}"
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Endpoint WebSocket: espera recibir un mensaje de texto con la imagen codificada en Base64,
    limpia el prefijo si es necesario, decodifica la imagen, procesa la detección y retorna
    el resultado junto con una descripción en lenguaje natural.
    """
    await websocket.accept()
    try:
        while True:
            # Recibir datos de la imagen en formato Base64
            data = await websocket.receive_text()
            data = clean_base64(data)

            try:
                image_bytes = base64.b64decode(data)
            except Exception as e:
                logger.error(f"Error al decodificar Base64: {e}")
                await websocket.send_text('{"error": "Error al decodificar la imagen"}')
                continue

            response = detect_objects(image_bytes)

                # Integrar OCR
            detected_text = extract_text_from_image(image_bytes)
             # Generar respuesta mejorada
            response["description"] = describe_scene(response.get("detected_objects", []), detected_text)

            if detected_text.strip():
                response["detected_text"] = detected_text

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("Cliente desconectado")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)