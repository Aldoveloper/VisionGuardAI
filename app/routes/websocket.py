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
from app.services.description_ai import generate_description
# filepath: c:\Users\Libardo Perdomo\VsGuardPy_v1\app\routes\websocket.py
from app.utils.commands import handle_command

# Configurar un caché en memoria con máximo 100 entradas y un TTL de 60 segundos.
cache = TTLCache(maxsize=100, ttl=60)
logger = logging.getLogger(__name__)

# Lista global para almacenar conexiones activas con estado
active_connections = {}

# Configurar un ThreadPoolExecutor para procesamiento paralelo (ajusta el número de hilos según tus recursos)
executor = ThreadPoolExecutor(max_workers=4)

def process_image(image_bytes: bytes) -> dict:
    """
    Esta función procesa la imagen:
      - Detecta objetos.
      - Extrae texto usando extract_text_from_image.(por ahora no se usa)
      - Genera la descripción final con describe_scene.
    Se ejecuta en un hilo separado para no bloquear el event loop.
    """
    # Detectar objetos en la imagen
    response = detect_objects(image_bytes)
    
    # Extraer texto de la imagen (en este ejemplo se simula)
    detected_text = "No hay Texto  detectado"
    
    # Generar la descripción combinando la detección y el texto extraído.
    response["description"] = generate_description(response.get("detected_objects", []),image_bytes)
    if detected_text.strip():
        response["detected_text"] = detected_text
    return response

async def send_safely(websocket: WebSocket, data: dict) -> bool:
    """Envía datos al WebSocket de manera segura, manejando posibles desconexiones."""
    try:
        await websocket.send_json(data)
        return True
    except Exception as e:
        logger.error(f"Error al enviar datos al WebSocket: {e}")
        return False

async def websocket_endpoint_SC(websocket: WebSocket):
    # Extraer client_id de los query params antes de aceptar la conexión.
    query_params = parse_qs(websocket.scope.get("query_string", b"").decode("utf-8"))
    client_id_list = query_params.get("client_id")
    logger.info(f"client_id recibido: {client_id_list}")
    if not client_id_list:
        logger.error("No se proporcionó client_id en la conexión.")
        await websocket.close()
        return
    client_id = client_id_list[0]

    await websocket.accept()
    # Agregar la conexión a la lista de conexiones del client_id.
    if client_id not in active_connections:
        active_connections[client_id] = []
    active_connections[client_id].append(websocket)
    logger.info(f"Nuevo cliente {client_id} conectado. Conexiones activas para este ID: {len(active_connections[client_id])}")

    try:
        while True:
             # Recibir datos de forma unificada
            message = await websocket.receive()
            logger.info(f"Datos recibidos: {message}")
            
            if "bytes" in message and message["bytes"] is not None:
                image_bytes = message["bytes"]
            elif "text" in message and message["text"] is not None:
                text_data = message["text"]

                # Si es un comando, procesarlo y omitir la lógica de imagen
                if await handle_command(websocket, text_data):
                    continue

                try:
                    image_bytes = base64.b64decode(text_data)
                except Exception as decode_err:
                    logger.error(f"Error decodificando base64: {decode_err}")
                    continue
            else:
                logger.error("No se recibieron datos válidos.")
                continue
            
            # Validar que image_bytes es efectivamente de tipo bytes
            if not isinstance(image_bytes, bytes):
                logger.error("El dato recibido no es del tipo esperado (bytes).")
                continue
            
            # Calcular un hash para la imagen recibida y usarlo para el sistema de caché.
            image_hash = hashlib.md5(image_bytes).hexdigest()

            if image_hash in cache:
                result = cache[image_hash]
            else:
                # Procesar la imagen en un hilo separado 
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(executor, process_image, image_bytes)
                cache[image_hash] = result
                           
            # Enviar la respuesta solo a las conexiones del client_id actual
            time_save_cache = time.perf_counter()
            send_count = 0
            logger.info(f"Enviando respuesta a {len(active_connections[client_id])} conexiones activas para client_id {client_id}.")

            if client_id in active_connections:
                # Lista para almacenar índices de conexiones fallidas
                to_remove = []
                for i, conn in enumerate(active_connections[client_id]):
                    success = await send_safely(conn, result)
                    if success:
                        send_count += 1
                    else:
                        to_remove.append(i)
                        
                # Eliminar conexiones fallidas (en orden inverso)
                for idx in sorted(to_remove, reverse=True):
                    active_connections[client_id].pop(idx)
                    
                # Si no quedan conexiones, eliminamos la clave
                if not active_connections[client_id]:
                    del active_connections[client_id]

            time_final_save_cache = time.perf_counter() - time_save_cache
            logger.info(f"Respuesta enviada a {send_count} conexiones para client_id {client_id} en: {time_final_save_cache:.2f} segundos.")
    except WebSocketDisconnect:
        logger.info(f"Cliente {client_id} desconectado.")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
    finally:
        # Asegurarse de que la conexión se elimine de la lista al finalizar
        if client_id in active_connections:
            # Encontrar y eliminar esta conexión específica
            if websocket in active_connections[client_id]:
                active_connections[client_id].remove(websocket)
                # Si no quedan conexiones para este cliente, eliminar la entrada
                if not active_connections[client_id]:
                    del active_connections[client_id]
            logger.info(f"Conexión del cliente {client_id} eliminada. Conexiones activas totales: {sum(len(c) for c in active_connections.values())}")