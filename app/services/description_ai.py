import logging
import base64
import tempfile
import time
import os
from dotenv import load_dotenv
load_dotenv()  # Carga las variables del .env

from google import genai
from app.utils.objeto_nombres import OBJETO_NOMBRES_ES

logger = logging.getLogger(__name__)

# Configura el cliente Gemini (ajusta tu API key)

gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

def build_scene_context(detected_objects: list) -> str:
    positions = {"izquierda": [], "centro": [], "derecha": []}
    for obj in detected_objects:
        label_es = OBJETO_NOMBRES_ES.get(obj["label"], obj["label"])
        color = obj.get("color")
        if color:
            label_es = f"{label_es} de color {color}"
        pos = obj.get("position", "centro")
        if pos in positions:
            positions[pos].append(label_es)
    
    context_parts = []
    if positions["izquierda"]:
        context_parts.append("A tu izquierda veo " + ", ".join(positions["izquierda"]) + ".")
    if positions["centro"]:
        context_parts.append("Frente a ti veo " + ", ".join(positions["centro"]) + ".")
    if positions["derecha"]:
        context_parts.append("A tu derecha veo " + ", ".join(positions["derecha"]) + ".")
    
    return " ".join(context_parts)

def build_spatial_context(detected_objects: list) -> str:
    positions = {"izquierda": [], "centro": [], "derecha": []}
    for obj in detected_objects:
        pos = obj.get("position", "centro")
        label_es = OBJETO_NOMBRES_ES.get(obj["label"], obj["label"])
        color = obj.get("color")
        if color:
            label_es = f"{label_es} de color {color}"
        if pos in positions:
            positions[pos].append(label_es)
    
    spatial_info = []
    if positions["centro"]:
        if positions["izquierda"]:
            spatial_info.append("se observa que hay objetos tanto en el centro como a la izquierda, lo que sugiere la posible presencia de obstáculos en esa zona")
        if positions["derecha"]:
            spatial_info.append("se detecta proximidad entre objetos en el centro y a la derecha, lo que podría dificultar una maniobra segura")
    elif positions["izquierda"] and positions["derecha"]:
        spatial_info.append("hay objetos en ambos laterales, por lo que se recomienda precaución al avanzar")
    
    if spatial_info:
        return " Además, " + " y ".join(spatial_info) + "."
    else:
        return ""
    

def generate_description(detected_objects: list, image_bytes: bytes) -> str:
    try:
        # Construir contextos de la escena y espacial
        scene_context = build_scene_context(detected_objects)
        spatial_context = build_spatial_context(detected_objects)
    
        # Formar el prompt final para Gemini
        prompt = (
            f"Describe de forma detallada pero corta, concisa y natural la siguiente escena, "
            f"comparando la información con la imagen y regresando la información verídica, "
            f"no agregues emoticones ni caracteres: {scene_context}{spatial_context}"
        )
        logger.info(f"Prompt para Gemini: {prompt}")

        # Guardar los bytes de la imagen en un archivo temporal.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            temp_filename = tmp.name

        # Subir el archivo usando client.files.upload
        my_file = client.files.upload(file=temp_filename)
        
        # Opcional: eliminar el archivo temporal si no se necesita
        os.remove(temp_filename)
        
        # Llamar a Gemini para generar la descripción.
        # Si la API de Gemini admite timeout, pásalo en los parámetros.
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[my_file, prompt],
        )
        
        # Validar la respuesta y devolver un fallback si es necesario.
        if not response.text or len(response.text.strip()) < 10:
            logger.warning("La respuesta de Gemini es insuficiente, usando fallback.")
            return "No se pudo generar una descripción fiable para la escena."
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error en generate_description: {e}")
        return "Error en la generación de descripción."