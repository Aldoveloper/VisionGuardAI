import base64
import logging
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocket, WebSocketDisconnect
from cachetools import TTLCache  # Librería para caching
from app.services.object_detection import detect_objects
from app.services.text_extraction import extract_text_from_image
from app.services.description import describe_scene

# Configurar un caché en memoria con máximo 100 entradas y un TTL de 60 segundos.
cache = TTLCache(maxsize=100, ttl=60)
logger = logging.getLogger(__name__)

# Configurar un ThreadPoolExecutor para procesamiento paralelo (ajusta el número de hilos según tus recursos)
executor = ThreadPoolExecutor(max_workers=4)

def process_image(image_bytes: bytes) -> dict:
    """
    Esta función procesa la imagen:
      - Detecta objetos.
      - Extrae texto usando extract_text_from_image.
      - Genera la descripción final con describe_scene.
    Se ejecuta en un hilo separado para no bloquear el event loop.
    """
    response = detect_objects(image_bytes)
    detected_text = extract_text_from_image(image_bytes)
    # Si extract_text_from_image devuelve una lista (por EasyOCR), la convertimos a string.
    if isinstance(detected_text, list):
        detected_text = " ".join([item[1] for item in detected_text if len(item) > 1])
    # Generar la descripción combinando la detección y el texto extraído.
    response["description"] = describe_scene(response.get("detected_objects", []), detected_text)
    if detected_text.strip():
        response["detected_text"] = detected_text
    return response

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Recibir datos de la imagen en base64 y limpiar el prefijo si existe.
            data = await websocket.receive_text()
            data = data.split(",")[-1]
            try:
                image_bytes = base64.b64decode(data)
            except Exception as e:
                logger.error(f"Error al decodificar Base64: {e}")
                await websocket.send_text('{"error": "Error al decodificar la imagen"}')
                continue

            # Calcular un hash para la imagen recibida y usarlo para el sistema de caché.
            image_hash = hashlib.md5(image_bytes).hexdigest()

            if image_hash in cache:
                result = cache[image_hash]
                logger.info("Resultado obtenido del caché.")
            else:
                # Ejecutar el procesamiento intensivo en un hilo separado para no bloquear.
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(executor, process_image, image_bytes)
                # Guardar el resultado en el caché.
                cache[image_hash] = result
                logger.info("Resultado procesado y guardado en caché.")

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("Cliente desconectado")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")