import hashlib, time, base64, asyncio, logging
from urllib.parse import parse_qs
from fastapi import WebSocket, WebSocketDisconnect
from app.models.response_model import DetectionResponse
from app.services.object_detection import detect_objects
from app.services.text_extraction import extract_text_from_image
from app.services.description_ai import generate_description
from app.utils.commands import handle_command

logger = logging.getLogger(__name__)
active_connections = {}  # Podrías incluso mover esto a un módulo de estado centralizado.
# Y define el ThreadPoolExecutor en este módulo o en un módulo de configuración:
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

async def websocket_endpoint(websocket: WebSocket):
    query_params = parse_qs(websocket.scope.get("query_string", b"").decode("utf-8"))
    client_id_list = query_params.get("client_id")
    logger.info(f"client_id recibido: {client_id_list}")
    if not client_id_list:
        logger.error("No se proporcionó client_id en la conexión.")
        await websocket.close()
        return
    client_id = client_id_list[0]

    await websocket.accept()
    if client_id not in active_connections:
        active_connections[client_id] = []
    active_connections[client_id].append(websocket)
    logger.info(f"Nuevo cliente {client_id} conectado. Conexiones activas: {len(active_connections[client_id])}")

    try:
        while True:
            message = await websocket.receive()
            logger.info(f"Datos recibidos: {message}")
            # Procesa mensaje según el tipo recibido
            if "bytes" in message and message["bytes"] is not None:
                image_bytes = message["bytes"]
            elif "text" in message and message["text"] is not None:
                text_data = message["text"]
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

            if not isinstance(image_bytes, bytes):
                logger.error("El dato recibido no es del tipo esperado (bytes).")
                continue

            image_hash = hashlib.md5(image_bytes).hexdigest()
            # Aquí puedes integrar un caché si es necesario.
            result = await asyncio.get_running_loop().run_in_executor(executor, process_image, image_bytes)

            # Enviar respuesta solo al mismo client_id
            send_count = 0
            for conn in active_connections.get(client_id, []):
                success = await send_safely(conn, result)
                if success:
                    send_count += 1
            logger.info(f"Respuesta enviada a {send_count} conexiones para client_id {client_id}.")
    except WebSocketDisconnect:
        logger.info(f"Cliente {client_id} desconectado.")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
    finally:
        if client_id in active_connections and websocket in active_connections[client_id]:
            active_connections[client_id].remove(websocket)
            if not active_connections[client_id]:
                del active_connections[client_id]
        logger.info(f"Conexión del cliente {client_id} eliminada.")
        
def process_image(image_bytes: bytes) -> dict:
    response = detect_objects(image_bytes)
    detected_text = "No hay Texto detectado"
    response["description"] = generate_description(response.get("detected_objects", []), image_bytes)
    if detected_text.strip():
        response["detected_text"] = detected_text
    return response

async def send_safely(websocket: WebSocket, data: dict) -> bool:
    try:
        await websocket.send_json(data)
        return True
    except Exception as e:
        logger.error(f"Error al enviar datos: {e}")
        return False