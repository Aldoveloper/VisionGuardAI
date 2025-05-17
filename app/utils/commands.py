# filepath: c:\Users\Libardo Perdomo\VsGuardPy_v1\app\utils\commands.py
import logging
from urllib.parse import parse_qs
from fastapi import WebSocket

logger = logging.getLogger(__name__)

async def handle_command(websocket: WebSocket, text: str) -> bool:
    """
    Procesa comandos enviados desde WebSocket.
    Devuelve True si el comando es reconocido y procesado.
    En este proceso, si el comando es "capture", se responde con "capture" a todas las conexiones
    que comparten el mismo client_id.
    """
    if text.strip().lower() == "capture":
        logger.info("Comando 'capture' recibido..")
        # Extraer el client_id de los query params del websocket
        qs = websocket.scope.get("query_string", b"").decode("utf-8")
        params = parse_qs(qs)
        client_id = params.get("client_id", [""])[0]
        # Realizar la importación de active_connections de forma diferida para evitar circularidad
        from app.routes.websocket import active_connections
        # Enviar el mensaje "capture" a todas las conexiones activas para ese client_id
        if client_id and client_id in active_connections:
            for conn in active_connections[client_id]:
                await conn.send_text("capture")
            logger.info(f"Se ha respondido 'capture' a {len(active_connections[client_id])} conexiones para client_id {client_id}.")
        else:
            logger.warning("No se encontró client_id en las conexiones activas.")
        return True
    return False