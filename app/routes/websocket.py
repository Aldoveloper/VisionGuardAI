import logging
import hashlib
import asyncio
import time
import base64
from urllib.parse import parse_qs


from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocket, WebSocketDisconnect
from cachetools import TTLCache  # Librería para caching
from app.services.object_detection import detect_objects
from app.services.text_extraction import extract_text_from_image
from app.services.description import describe_scene

# Configurar un caché en memoria con máximo 100 entradas y un TTL de 60 segundos.
cache = TTLCache(maxsize=100, ttl=60)
logger = logging.getLogger(__name__)

# Lista global para almacenar conexiones activas
active_connections = []

# Configurar un ThreadPoolExecutor para procesamiento paralelo (ajusta el número de hilos según tus recursos)
executor = ThreadPoolExecutor(max_workers=4)

def process_image(image_bytes: bytes) -> dict:
    inicia_proceso_image = time.perf_counter()
    """
    Esta función procesa la imagen:
      - Detecta objetos.
      - Extrae texto usando extract_text_from_image.
      - Genera la descripción final con describe_scene.
    Se ejecuta en un hilo separado para no bloquear el event loop.
    """
    # Detectar objetos en la imagen
    response = detect_objects(image_bytes)
    
    # Extraer texto de la imagen (en este ejemplo se simula)
    detected_text = "Texto detectado"
    
    # Generar la descripción combinando la detección y el texto extraído.
    response["description"] = describe_scene(response.get("detected_objects", []), detected_text)
    if detected_text.strip():
        response["detected_text"] = detected_text

    final_time_proceso_image = time.perf_counter() - inicia_proceso_image
    logger.info(f"Tiempo total para procesar la imagen: {final_time_proceso_image:.2f} segundos.")
    return response

async def websocket_endpoint(websocket: WebSocket):
    # Extraer client_id de los query params antes de aceptar la conexión.
    query_params = parse_qs(websocket.scope.get("query_string", b"").decode("utf-8"))
    client_id_list = query_params.get("client_id")
    if not client_id_list:
        logger.error("No se proporcionó client_id en la conexión.")
        await websocket.close()
        return
    client_id = client_id_list[0]

    await websocket.accept()
    active_connections.append(websocket)
     # Agregar la conexión a la lista de conexiones del client_id.
    # if client_id not in active_connections:
    #     active_connections[client_id] = []
    # active_connections[client_id].append(websocket)
    # logger.info(f"Nuevo cliente {client_id} conectado. Conexiones activas para este ID: {len(active_connections[client_id])}")

    try:
        while True:
            # Recibir datos: se intenta primero en binario; si falla, se intenta recibir texto y decodificar base64.
            try:
                try:
                    image_bytes = await websocket.receive_bytes()
                except Exception as e:
                    text_data = await websocket.receive_text()
                    logger.info("Datos recibidos en formato texto.", text_data)
                    try:
                        image_bytes = base64.b64decode(text_data)
                    except Exception as decode_err:
                        logger.error(f"Error decodificando base64: {decode_err}")
                        await websocket.send_json({
                            "detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}],
                            "error": "Error en la decodificación de datos base64"
                        })
                        continue
            except Exception as e:
                logger.error(f"Error al recibir datos: {e}")
                await websocket.send_text('{"error": "Error en la recepción de datos"}')
                break
            
            # Validar que image_bytes es efectivamente de tipo bytes
            if not isinstance(image_bytes, bytes):
                logger.error("El dato recibido no es del tipo esperado (bytes).")
                await websocket.send_json({
                    "detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}],
                    "error": "Formato de datos inválido"
                })
                continue
            
            # Calcular un hash para la imagen recibida y usarlo para el sistema de caché.
            image_hash = hashlib.md5(image_bytes).hexdigest()

            if image_hash in cache:
                result = cache[image_hash]
                logger.info("Resultado obtenido del caché.")
            else:
                # Procesar la imagen en un hilo separado 
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(executor, process_image, image_bytes)
                cache[image_hash] = result
                           
            # Enviar la respuesta a todos los clientes conectados
            time_save_cache = time.perf_counter()
            for connection in active_connections:
                await connection.send_json(result)
            time_final_save_cache = time.perf_counter() - time_save_cache
            logger.info(f"Respuesta enviada a {len(active_connections)} clientes en: {time_final_save_cache:.2f} segundos.")

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Cliente desconectado. Conexiones activas: {len(active_connections)}")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        await websocket.send_json({
            "detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}],
            "error": "Error en detección"
        })