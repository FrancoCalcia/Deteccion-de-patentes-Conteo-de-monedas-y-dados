import cv2
import matplotlib.pyplot as plt
import numpy as np

def cargar_imagen(image_path):
    """Carga una imagen desde una ruta proporcionada."""
    return cv2.imread(image_path)

def convertir_a_grises(image):
    """Convierte una imagen a escala de grises si no está ya en escala de grises."""
    if len(image.shape) == 3 and image.shape[2] == 3:  # Verifica si la imagen tiene 3 canales
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Retorna la imagen sin cambios si ya está en escala de grises


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

def clasificar_y_calcular_fp(recorte_procesado):
    """Clasifica el recorte procesado en moneda o dado y muestra el factor de forma (Fp) de cada contorno."""
    
    # Encontrar contornos en la imagen procesada
    contours, _ = cv2.findContours(recorte_procesado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lista para almacenar los factores de forma de cada contorno
    fps = []
    
    # Iterar sobre los contornos detectados
    for contour in contours:
        # Calcular el área y el perímetro del contorno
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Evitar divisiones por cero
        if perimeter == 0:
            continue
        
        # Calcular el factor de forma (Fp)
        factor_forma = area / (perimeter ** 2)
        fps.append(factor_forma)
        
        # Clasificación basada en el factor de forma
        if factor_forma > 0.07:  # Umbral para objetos circulares (monedas)
            tipo_objeto = "Moneda"
        else:
            tipo_objeto = "Dado"
        
        # Mostrar el resultado
        print(f"Factor de Forma (Fp) para el objeto: {factor_forma:.4f} - Clasificación: {tipo_objeto}")
    
    return fps

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
def aplicar_apertura_clausura(image, kernel_size=(5, 5)):
    """Aplica apertura seguida de clausura a la imagen para mejorar la calidad de los bordes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Aplicar apertura (reduce el ruido)
    apertura = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Aplicar clausura (cierra pequeños huecos en los bordes)
    clausura = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)
    
    return clausura

def contar_regiones_internas(image):
    """Cuenta las regiones internas en un dado."""
    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Contar solo los contornos internos (hijos)
    return sum(1 for i in range(len(contours)) if _[0, i, 3] != -1)

def clasificar_recorte(recorte, index):
    """Clasifica el recorte como moneda o dado y cuenta las regiones internas si es un dado."""
    # Convertir el recorte a escala de grises y aplicar binarización
    gray = convertir_a_grises(recorte)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Aplicar dilatación para conectar puntos dispersos en el caso de los dados
    thresh_dilated = cv2.dilate(thresh, None, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"Recorte {index + 1}: No se detectaron contornos.")
        return "Desconocido"
    
    # Clasificar usando el factor de forma (Fp)
    tipo_objeto = clasificar_y_calcular_fp(thresh_dilated)
    
    # Clasificación de dado: contar las regiones internas
    if tipo_objeto == "Dado":
        regiones_internas = contar_regiones_internas(thresh)
        print(f"Recorte {index + 1}: Clasificación: {tipo_objeto}, Regiones internas: {regiones_internas}")
    else:
        print(f"Recorte {index + 1}: Clasificación: {tipo_objeto}")
    
    return tipo_objeto

# Código de procesamiento de la imagen y detección de bordes (sin cambios)
image_path = "archivos/monedas.jpg"
image = cargar_imagen(image_path)
gray_image = convertir_a_grises(image)
blurred_image = aplicar_desenfoque(gray_image)
edges = detectar_bordes(blurred_image)
dilated_edges = dilatar_bordes(edges)

# Aplicar apertura y clausura
processed_edges = aplicar_apertura_clausura(dilated_edges)

# Mostrar las etapas de procesamiento
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title("Imagen Original")
ax[0].axis("off")

ax[1].imshow(gray_image, cmap='gray')
ax[1].set_title("Imagen en Escala de Grises")
ax[1].axis("off")

ax[2].imshow(edges, cmap='gray')
ax[2].set_title("Bordes Detectados (Canny)")
ax[2].axis("off")

ax[3].imshow(processed_edges, cmap='gray')
ax[3].set_title("Bordes con Apertura y Clausura")
ax[3].axis("off")

plt.tight_layout()
plt.show()

# Detectar y dibujar contornos en la imagen procesada
contours = detectar_contornos(processed_edges)
contour_image = dibujar_contornos(processed_edges, contours)
mostrar_contornos(contour_image)

# Recortar y mostrar cada contorno
recortes = recortar_contornos(processed_edges, contours)

# Mostrar el número total de objetos detectados
print(f"Número total de objetos detectados: {len(contours)}")

# Clasificar cada recorte
for i, recorte in enumerate(recortes):
    tipo_objeto = clasificar_recorte(recorte, i)
    
    if tipo_objeto == "Moneda":
        tipo_moneda = distinguir_moneda(recorte)
        print(f"Objeto {i + 1}: {tipo_objeto} - {tipo_moneda}")
    elif tipo_objeto == "Dado":
        print(f"Objeto {i + 1}: {tipo_objeto}")
    else:
        print(f"Objeto {i + 1}: No clasificado")


# 11 y 19 son los dados