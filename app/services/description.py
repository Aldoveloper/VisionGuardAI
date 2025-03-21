from collections import Counter
from app.utils.objeto_nombres import OBJETO_NOMBRES_ES
from app.services.text_extraction import extract_text_from_image, detect_text_noise


def format_objects(obj_list):
    obj_count = Counter(obj_list)
    formatted_list = []

    for obj, count in obj_count.items():
        obj_es = OBJETO_NOMBRES_ES.get(obj, obj)  # Traducción al español
        obj_plural = obj_es + "s" if obj_es.endswith(("a", "e", "o")) else obj_es
        article = "una" if obj_es.endswith("a") else "un"

        if obj_es == "persona":
            formatted_list.append(f"a {article} {obj_es}" if count == 1 else f"a {count} {obj_plural}")
        else:
            formatted_list.append(f"{article} {obj_es}" if count == 1 else f"{count} {obj_plural}")

    return ", ".join(formatted_list[:-1]) + " y " + formatted_list[-1] if len(formatted_list) > 1 else formatted_list[0] if formatted_list else ""

def describe_scene(detected_objects: list, detected_text: str = "") -> str:
    if not detected_objects and not detected_text:
        return "No se detectaron objetos ni texto en la imagen."

    description = ""
    locations = {"izquierda": [], "centro": [], "derecha": []}

    for obj in detected_objects:
        label_es = OBJETO_NOMBRES_ES.get(obj["label"], obj["label"])
        locations[obj["position"]].append(label_es)

    if locations["izquierda"]:
        description += f"A tu izquierda veo {format_objects(locations['izquierda'])}. "
    if locations["centro"]:
        description += f"Frente a ti {format_objects(locations['centro'])}. "
    if locations["derecha"]:
        description += f"A tu derecha {format_objects(locations['derecha'])}. "

    if any(obj["label"] in {"car", "person", "bicycle", "motorcycle", "truck", "bus"} for obj in detected_objects):
        description += "Ten cuidado, hay obstáculos en tu camino. "

    corrected_text = detect_text_noise(detected_text)
    if corrected_text:
        if corrected_text.get("noMessage"):
            description += f"También veo un letrero o texto,'{corrected_text.get('message')}'. "
        else:
            description += f"También veo un letrero o texto que dice: '{corrected_text.get('message')}'. "

   
    return description.strip()
