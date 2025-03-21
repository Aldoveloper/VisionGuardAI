import numpy as np
import cv2
from app.config import model, logger

def detect_objects(image_bytes: bytes) -> dict:
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("La imagen no se pudo decodificar.")
            return {"error": "No se pudo decodificar la imagen"}

        height, width, _ = frame.shape
        results = model(frame)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names.get(cls, "desconocido")

                x_center, _, _, _ = box.xywh[0]
                position = "izquierda" if x_center < width / 3 else "derecha" if x_center > 2 * width / 3 else "centro"

                detected_objects.append({"label": label, "position": position})

        return {"detected_objects": detected_objects}

    except Exception as e:
        logger.error(f"Error en detect_objects: {e}")
        return {"error": "Error en detección"}
