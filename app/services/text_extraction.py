import numpy as np
import cv2
import pytesseract
import easyocr
import re

reader = easyocr.Reader(['es', 'en'])

def extract_text_from_image(image_bytes: bytes, use_easyocr=True) -> str:
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return "No se pudo procesar la imagen."

        return " ".join(reader.readtext(img, detail=0)) if use_easyocr else pytesseract.image_to_string(img, lang='spa+eng')

    except Exception as e:
        return f"Error en OCR: {e}"

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
                return {"noMessage": True, "message": "Pero el texto no es completamente legible."}


    return detected_text
