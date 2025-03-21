import base64
import os

def image_to_base64(image_path):
    """Convierte una imagen local a una cadena Base64."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        print("Error: Archivo no encontrado.")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    downloads_folder = os.path.expanduser("~/Downloads")  # Obtiene la ruta de Descargas
    image_name = "t2.jpg"  # Reemplaza con el nombre del archivo en Descargas
    image_path = os.path.join(downloads_folder, image_name)
    
    base64_string = image_to_base64(image_path)
    if base64_string:
        print("Imagen en Base64:", base64_string)  # Imprime solo los primeros 100 caracteres
        print("Longitud de la cadena Base64:", len(base64_string))