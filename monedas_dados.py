import cv2
import matplotlib.pyplot as plt
import numpy as np

def cargar_imagen(image_path):
    """Carga una imagen desde una ruta proporcionada."""
    return cv2.imread(image_path)

def convertir_a_grises(image):
    """Convierte una imagen a escala de grises."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def aplicar_desenfoque(image, kernel_size=(5, 5)):
    """Aplica un desenfoque Gaussiano a la imagen."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def detectar_bordes(image, low_threshold=80, high_threshold=180):
    """Detecta bordes en una imagen utilizando el algoritmo de Canny."""
    return cv2.Canny(image, low_threshold, high_threshold)

def dilatar_bordes(image, iterations=15):
    """Dilata los bordes para unir fragmentos cercanos."""
    return cv2.dilate(image, None, iterations=iterations)

def mostrar_imagenes(image_original, gray_image, edges):
    """Muestra la imagen original, en escala de grises y los bordes detectados."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    ax[1].imshow(gray_image, cmap='gray')
    ax[1].set_title("Imagen en Escala de Grises")
    ax[1].axis("off")

    ax[2].imshow(edges, cmap='gray')
    ax[2].set_title("Bordes Detectados (Canny)")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

def detectar_contornos(image):
    """Encuentra contornos en la imagen y devuelve la lista de contornos."""
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def dibujar_contornos(image, contours):
    """Dibuja los contornos detectados sobre una copia de la imagen original."""
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image

def mostrar_contornos(contour_image):
    """Muestra la imagen con los contornos dibujados."""
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title("Contornos Detectados")
    plt.axis("off")
    plt.show()

def recortar_contornos(image, contours):
    """Recorta cada contorno detectado y guarda los recortes en una lista."""
    recortes = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        recorte = image[y:y+h, x:x+w]
        recortes.append(recorte)
        
        # Mostrar cada recorte
        plt.figure()
        plt.imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
        plt.title(f"Recorte del Contorno {i + 1}")
        plt.axis("off")
        plt.show()
    return recortes

def clasificar(recorte):
    """Clasifica el recorte en moneda o dado usando una aproximación basada en el análisis de bordes."""
    # Convertir recorte a escala de grises y aplicar desenfoque
    gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes
    edges = cv2.Canny(blurred, 50, 150)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contar el número de lados aproximados del contorno
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # Usar el número de lados para identificar dado (cuadrado) o moneda (circular)
        if len(approx) == 4:  # El contorno es cuadrado
            return "Dado"
        else:  # El contorno es redondeado
            return "Moneda"
    return "Indeterminado"

def distinguir_moneda(recorte_moneda):
    """Distingue el tipo de moneda por tamaño usando el área del contorno."""
    # Convertir a escala de grises
    gray = cv2.cvtColor(recorte_moneda, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar contornos
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcular el área del contorno más grande (debería ser el de la moneda)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
    
    # Clasificar según el área (umbral de ejemplo)
    if max_area > 2000:
        return "Moneda Grande"
    elif max_area > 1000:
        return "Moneda Mediana"
    else:
        return "Moneda Pequeña"

# Cargar y procesar la imagen
image_path = "archivos/monedas.jpg"
image = cargar_imagen(image_path)
gray_image = convertir_a_grises(image)
blurred_image = aplicar_desenfoque(gray_image)
edges = detectar_bordes(blurred_image)
dilated_edges = dilatar_bordes(edges)

# Mostrar etapas de procesamiento
mostrar_imagenes(image, gray_image, edges)

# Detectar y dibujar contornos
contours = detectar_contornos(dilated_edges)
contour_image = dibujar_contornos(image, contours)
mostrar_contornos(contour_image)

# Recortar y mostrar cada contorno
recortes = recortar_contornos(image, contours)

# Mostrar el número total de objetos detectados
print(f"Número total de objetos detectados: {len(contours)}")

for i, recorte in enumerate(recortes):
    tipo_objeto = clasificar(recorte)
    
    if tipo_objeto == "Moneda":
        tipo_moneda = distinguir_moneda(recorte)
        print(f"Objeto {i + 1}: {tipo_objeto} - {tipo_moneda}")
    elif tipo_objeto == "Dado":
        print(f"Objeto {i + 1}: {tipo_objeto}")
    else:
        print(f"Objeto {i + 1}: No clasificado")

# 11 y 19 son los dados