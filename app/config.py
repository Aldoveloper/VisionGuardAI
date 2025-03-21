import logging
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar modelo
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error al cargar YOLO: {e}")
