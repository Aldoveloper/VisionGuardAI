import numpy as np
import cv2
from app.config import model, logger
import time
import os

def detect_objects(image_bytes: bytes, conf_threshold: float = 0.2) -> dict:
    """
    Detecta objetos en la imagen y calcula el color promedio de cada objeto detectado.
    """
    try:
        if not image_bytes:
            logger.error("No se recibió ningún dato de imagen.")
            return {"detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}]}

        # Log de verificación: longitud de los datos recibidos
        
        # Opcional: guardar la imagen a disco para depuración
        temp_path = "temp_debug_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        # Convertir los bytes de la imagen a un array de Numpy y decodificarla
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("La imagen no se pudo decodificar.")
            return {"detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}]}
        
        height, width, _ = frame.shape
        # Enviar el frame al modelo para detección
        results = model(frame)
        
        detected_objects = []
        for result in results:
            for box in result.boxes:
                # Extraer la confianza de la detección
                confidence = float(box.conf[0])
                if confidence < conf_threshold:
                    continue
                
                # Extraer la clase y obtener la etiqueta correspondiente
                cls = int(box.cls[0])
                label = model.names.get(cls, "desconocido")
                
                # Calcular la posición del objeto según el centro de la caja (box.xywh)
                x_center, y_center, w, h = box.xywh[0]
                position = (
                    "izquierda" if x_center < width / 3
                    else "derecha" if x_center > 2 * width / 3
                    else "centro"
                )
                
                # Calcular coordenadas de la región de interés (ROI)
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                
                # Asegurarse de que las coordenadas estén dentro de los límites
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    color_str = "desconocido"
                else:
                    # Calcular el color promedio en formato BGR y convertir a RGB
                    avg_color_bgr = cv2.mean(roi)[:3]
                    avg_color_rgb = (int(avg_color_bgr[2]), int(avg_color_bgr[1]), int(avg_color_bgr[0]))
                    color_str = f"{avg_color_rgb}"
                
                detected_objects.append({
                    "label": label,
                    "position": position,
                    "confidence": confidence,
                    "color": color_str
                })
        
        if not detected_objects:
            return {"detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}]}
        
        logger.info(f"Objetos detectados: {detected_objects}")
        return {"detected_objects": detected_objects}

    except Exception as e:
        logger.error(f"Error en detect_objects: {e}")
        return {
            "detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}],
            "error": "Error en detección"
        }