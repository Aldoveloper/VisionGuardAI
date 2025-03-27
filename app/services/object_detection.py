import numpy as np
import cv2
from app.config import model, logger
import time

def detect_objects(image_bytes: bytes, conf_threshold: float = 0.2) -> dict:
    """
    Detecta objetos en la imagen.
    
    Parámetros:
      - image_bytes: Bytes de la imagen a procesar.
      - conf_threshold: Umbral mínimo de confianza para aceptar una detección.
    
    Retorna:
      Diccionario con las detecciones. Si ocurre algún error o no se detecta nada,
      se retorna un objeto por defecto:
      {"detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}]}
    """
    try:
        # Convertir los bytes de la imagen a un array de Numpy y decodificarla
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("La imagen no se pudo decodificar.")
            return {"detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}]}
        
        logger.info("Imagen recibida y decodificada correctamente.")
       
        height, width, _ = frame.shape
        # Enviar el frame al modelo para detección
        medir_tiempo_proceso = time.perf_counter()
        results = model(frame)
        final_time_process_image = time.perf_counter() - medir_tiempo_proceso
        logger.info(f"Tiempo para procesar la imagen: {final_time_process_image:.2f} segundos.")

        detected_objects = []
        
        for result in results:
            for box in result.boxes:
                # Extraer la confianza de la detección
                confidence = float(box.conf[0])
                # Sólo se agrega el objeto si la confianza es mayor o igual que el umbral
                if confidence < conf_threshold:
                    continue
                
                # Extraer la clase y obtener la etiqueta correspondiente
                cls = int(box.cls[0])
                label = model.names.get(cls, "desconocido")
                
                # Calcular la posición del objeto según el centro de la caja (usando box.xywh)
                x_center, y_center, w, h = box.xywh[0]
                position = (
                    "izquierda" if x_center < width / 3
                    else "derecha" if x_center > 2 * width / 3
                    else "centro"
                )
                
                detected_objects.append({
                    "label": label,
                    "position": position,
                    "confidence": confidence
                })
        
        if not detected_objects:
            # Si no se detecta nada, retornar un objeto por defecto
            detected_objects = [{"label": "desconocido", "position": "desconocida", "confidence": 0}]
        
        return {"detected_objects": detected_objects}

    except Exception as e:
        logger.error(f"Error en detect_objects: {e}")
        # En caso de error, retorna el objeto por defecto indicando que no se obtuvo detección
        return {"detected_objects": [{"label": "desconocido", "position": "desconocida", "confidence": 0}], "error": "Error en detección"}