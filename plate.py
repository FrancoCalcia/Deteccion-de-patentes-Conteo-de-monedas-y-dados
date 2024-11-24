import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar imágenes usando Matplotlib
# Esta función permite visualizar imágenes en escala de grises o en color.
def imshow(img, title=None, color_img=False, colorbar=True, ticks=False):
    """
    Muestra una imagen con título y opciones de visualización.

    Args:
        img (numpy.ndarray): La imagen a mostrar.
        title (str): Título de la imagen.
        color_img (bool): Indica si la imagen es en color (True) o en escala de grises (False).
        colorbar (bool): Si se debe mostrar una barra de color.
        ticks (bool): Si se deben mostrar los ejes de la imagen.
    """
    plt.figure()
    if color_img:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir de BGR a RGB para visualizar correctamente
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    plt.show()

# Función para preprocesar una imagen
def preprocesar_imagen(img, threshold=63, canny_min=200, canny_max=350, kernel_size=(9, 4), iterations=2):
    """
    Preprocesa una imagen para detectar bordes y aplicar operaciones morfológicas.

    Args:
        img (numpy.ndarray): Imagen original en color.
        threshold (int): Valor de umbral para binarización.
        canny_min (int): Valor mínimo del umbral para el detector de bordes Canny.
        canny_max (int): Valor máximo del umbral para el detector de bordes Canny.
        kernel_size (tuple): Tamaño del elemento estructurante para la morfología.
        iterations (int): Cantidad de iteraciones para la operación de cierre.

    Returns:
        numpy.ndarray: Imagen preprocesada con cierre aplicado.
    """
    # Mostrar la imagen original
    imshow(img, title="Imagen RGB")

    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imshow(img_gray, title="Imagen en blanco y negro")

    # Aplicar binarización usando un umbral fijo
    _, img_bin = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    imshow(img_bin, title="Imagen binarizada")

    # Aplicar detector de bordes (Canny)
    img_canny = cv2.Canny(img_bin, canny_min, canny_max)
    imshow(img_canny, title="Imagen luego de aplicar Canny")

    # Aplicar operación morfológica de cierre
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    img_close = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    imshow(img_close, title="Imagen despues de aplicar morfología (cierre)")

    return img_close

# Función para filtrar componentes conectadas con base en su área
def filtrar_componentes_conectadas(img, min_area=1500):
    """
    Filtra componentes conectadas de una imagen binaria según el área mínima.

    Args:
        img (numpy.ndarray): Imagen binaria.
        min_area (int): Área mínima requerida para conservar un componente.

    Returns:
        numpy.ndarray: Imagen con componentes conectadas filtradas.
    """
    # Detectar componentes conectadas y sus estadísticas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Crear una nueva imagen para almacenar los componentes filtrados
    filtered_img = np.zeros_like(img, dtype=np.uint8)

    # Iterar sobre los componentes detectados (excepto el fondo)
    for i in range(1, num_labels):  # La etiqueta 0 corresponde al fondo
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            # Conservar componentes que superen el área mínima
            filtered_img[labels == i] = 255

    imshow(filtered_img, title="Componentes conectadas filtradas")
    return filtered_img

# Función para detectar posibles patentes en una imagen binaria
def encontrar_patentes(img_bin, img, rect_min_size=(42, 11), rect_max_size=(103, 46), epsilon_factor=0.0345, show_result=True):
    """
    Detecta regiones de posibles patentes en la imagen binaria y las recorta.

    Args:
        img_bin (numpy.ndarray): Imagen binaria después del preprocesamiento.
        img (numpy.ndarray): Imagen original en color.
        rect_min_size (tuple): Dimensiones mínimas del rectángulo.
        rect_max_size (tuple): Dimensiones máximas del rectángulo.
        epsilon_factor (float): Factor para aproximación de polígonos.
        show_result (bool): Si se debe mostrar la imagen con las patentes detectadas.

    Returns:
        list: Lista de imágenes de posibles patentes recortadas.
    """
    # Encontrar contornos externos
    ext_contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posibles_patentes = []  # Lista para almacenar las posibles patentes
    img_out = img.copy()  # Copia de la imagen original para visualizar resultados

    # Obtener las dimensiones de la imagen
    height, width = img.shape[:2]

    for contour in ext_contours:
        # Aproximar el contorno a un polígono
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Verificar si el polígono tiene 4 lados y cumple con las dimensiones esperadas
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h  # Relación de aspecto del rectángulo

        if (len(approx) == 4 and
            rect_min_size[0] <= w <= rect_max_size[0] and
            rect_min_size[1] <= h <= rect_max_size[1] and
            2.1 < aspect_ratio < 3.4):  # Validar relación de aspecto
            # Calcular un margen alrededor de la patente
            margin_x = int(w * 0.2)  # Margen horizontal
            margin_y = int(h * 0.5)  # Margen vertical

            # Ajustar los límites del rectángulo considerando el margen
            x_start = max(0, x - margin_x)
            x_end = min(width, x + w + margin_x)
            y_start = max(0, y - margin_y)
            y_end = min(height, y + h + margin_y)

            # Recortar la región correspondiente a la posible patente
            patente = img[y_start:y_end, x_start:x_end]
            posibles_patentes.append(patente)  # Agregar a la lista

            # Dibujar el contorno en la imagen original
            cv2.drawContours(img_out, [contour], -1, (0, 255, 0), 2)

    if show_result:
        imshow(img_out, title="Patentes detectadas", color_img=True)

    return posibles_patentes


def encontrar_patentes_04(img_bin, img, rect_min_size=(30, 15), rect_max_size=(600, 600), show_result=True):
    """
    Detecta regiones de posibles patentes en la imagen binaria y las recorta.

    Args:
        img_bin (numpy.ndarray): Imagen binaria después del preprocesamiento.
        img (numpy.ndarray): Imagen original en color.
        rect_min_size (tuple): Dimensiones mínimas del rectángulo.
        rect_max_size (tuple): Dimensiones máximas del rectángulo.
        epsilon_factor (float): Factor para aproximación de polígonos.
        show_result (bool): Si se debe mostrar la imagen con las patentes detectadas.

    Returns:
        list: Lista de imágenes de posibles patentes recortadas.
    """
    # Unir contornos cercanos mediante una operación morfológica de cierre
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))  # Ajusta el tamaño del kernel según tus necesidades
    img_closed = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos externos en la imagen "cerrada"
    ext_contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posibles_patentes = []  # Lista para almacenar las posibles patentes
    img_out = img.copy()  # Copia de la imagen original para visualizar resultados

    # Obtener las dimensiones de la imagen
    height, width = img.shape[:2]

    for contour in ext_contours:
        # Verificar si el polígono tiene 4 lados y cumple con las dimensiones esperadas
        x, y, w, h = cv2.boundingRect(contour)

        if (rect_min_size[0] <= w <= rect_max_size[0] and
            rect_min_size[1] <= h <= rect_max_size[1]):
            # Calcular un margen alrededor de la patente
            margin_x = int(w * 0.2)  # Margen horizontal
            margin_y = int(h * 0.5)  # Margen vertical

            # Ajustar los límites del rectángulo considerando el margen
            x_start = max(0, x - margin_x)
            x_end = min(width, x + w + margin_x)
            y_start = max(0, y - margin_y)
            y_end = min(height, y + h + margin_y)

            # Recortar la región correspondiente a la posible patente
            patente = img[y_start:y_end, x_start:x_end]
            posibles_patentes.append(patente)  # Agregar a la lista

            # Dibujar el contorno en la imagen original
            cv2.drawContours(img_out, [contour], -1, (0, 255, 0), 2)

    if show_result:
        imshow(img_out, title="Patentes detectadas", color_img=True)

    return posibles_patentes


# Función para recortar la primera patente detectada de una lista de posibles patentes
def recortar_primer_patente(posibles_patentes):
    """
    Recorta la primera patente de la lista de posibles patentes.

    Args:
        posibles_patentes (list): Lista de imágenes de posibles patentes.

    Returns:
        numpy.ndarray: Imagen en escala de grises de la primera patente recortada, o None si no se encontró ninguna patente.
    """
    if len(posibles_patentes) == 0:
        print("No se detectaron patentes para recortar.")
        return None

    # Procesar solo la primera patente detectada
    primera_patente = posibles_patentes[0]
    patente_gray = cv2.cvtColor(primera_patente, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

    # Mostrar la patente recortada
    imshow(patente_gray, title="Primer patente recortada")

    return patente_gray

# Función para preprocesar una patente antes de segmentar las letras
def preprocesar_patente(gris):
    """
    Realiza el preprocesamiento para resaltar las letras en una patente.

    Args:
        gris (numpy.ndarray): Imagen de la patente en escala de grises.

    Returns:
        numpy.ndarray: Imagen binarizada después de aplicar un filtro Black Hat.
    """
    # Crear un elemento estructurante grande para la operación Black Hat
    black_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (76, 36))

    # Aplicar el filtro Black Hat para resaltar las letras (fondo oscuro, letras claras)
    img_black = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, black_hat, iterations=2)

    # Binarizar la imagen resultante
    _, bini = cv2.threshold(img_black, 70, 255, cv2.THRESH_BINARY)
    imshow(bini, title="Patente después de Black Hat")

    return bini

# Función para segmentar y procesar las letras de una patente
def procesar_patente(gris):
    """
    Preprocesa, binariza y segmenta las letras en una patente.

    Args:
        gris (numpy.ndarray): Imagen en escala de grises de una patente.

    Returns:
        list: Lista de tuplas con las letras recortadas y sus áreas.
    """
    # Paso 1: Preprocesar la imagen
    preprocesada = preprocesar_patente(gris)

    # Paso 2: Binarizar la imagen
    _, patente_bin = cv2.threshold(preprocesada, 10, 255, cv2.THRESH_BINARY_INV)
    imshow(patente_bin, title="Patente binarizada")

    # Paso 3: Segmentación de componentes conectadas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(patente_bin, connectivity=4)

    # Lista para almacenar las letras recortadas
    letras = []

    # Iterar sobre cada componente conectado (excepto el fondo)
    for i in range(1, num_labels):  # La etiqueta 0 es el fondo
        x, y, w, h, area = stats[i]
        if 16 < area < 125:  # Ajustar el filtro según el tamaño esperado de las letras
            letra = patente_bin[y:y+h, x:x+w]
            letras.append((x, letra, area))  # Guardar la letra y su área

    # Ordenar las letras de izquierda a derecha por sus posiciones horizontales
    letras_ordenadas = [(letra, area) for _, letra, area in sorted(letras, key=lambda item: item[0])]

    return letras_ordenadas

# Función principal que ejecuta el procesamiento completo
def main():
    """
    Función principal que procesa varias imágenes de patentes.
    Para cada imagen, realiza el preprocesamiento, detecta posibles patentes,
    recorta la primera patente y segmenta sus letras.
    """
    # Lista de identificadores de las imágenes
    ids = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    
    global imagen_a_reevaluar  # Lista para guardar imágenes no detectadas
    imagen_a_reevaluar = [] 
    
    for id in ids:
        img_path = f"archivos/img{id}.png"  # Ruta de la imagen
        img = cv2.imread(img_path)  # Cargar la imagen
        if img is None:
            print(f"Error: No se pudo cargar la imagen desde '{img_path}'")
            continue

        # Mostrar la imagen original
        imshow(img, title="Imagen original", color_img=True)

        # Preprocesar la imagen
        img_close = preprocesar_imagen(img)

        # Filtrar componentes conectadas según el área
        filtered_img = filtrar_componentes_conectadas(img_close)

        # Refinar la imagen con una operación de apertura
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 4))
        img_open = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel, iterations=3)
        imshow(img_open, title="Refinamiento: Apertura")

        # Detectar posibles patentes en la imagen refinada
        posibles_patentes = encontrar_patentes(img_open, img)
        if len(posibles_patentes) == 0:
            print("No se encontraron posibles patentes. Guardando la imagen para reevaluar...")
            # Guarda la imagen original para reevaluarla
            nombre_archivo = f"archivos/img{id}_no_detectada.png"
            cv2.imwrite(nombre_archivo, img)
            print(f"Imagen guardada como: {nombre_archivo}")
            imagen_a_reevaluar.append(nombre_archivo)
        else:
            print(f"Se encontraron {len(posibles_patentes)} posibles patentes.")

        # Si se detectaron patentes, procesar la primera patente
        if posibles_patentes:
            primera_patente = posibles_patentes[0]  # Obtener la primera patente detectada
            imshow(primera_patente, title="Patente a color RGB")

            # Convertir la patente a escala de grises y mostrarla
            primera_patente_gray = cv2.cvtColor(primera_patente, cv2.COLOR_BGR2GRAY)
            imshow(primera_patente_gray, title="Patente recortada en blanco y negro")

            # Procesar las letras de la patente
            letras_recortadas = procesar_patente(primera_patente_gray)
            print(f"Se encontraron {len(letras_recortadas)} letras en la primera patente.")

            # Mostrar y dilatar las letras encontradas
            for i, (letra, area) in enumerate(letras_recortadas):
                print(f"Letra {i + 1}: Área = {area}")
                if 16 < area < 125:  # Validar el área de la letra
                    # Aplicar dilatación para mejorar la visibilidad de la letra
                    kernel = np.ones((2, 1), np.uint8)
                    letra_dilatada = cv2.dilate(letra, kernel, iterations=1)
                    imshow(letra_dilatada, title=f"Letra {i + 1}")
                else:
                    print(f"Letra {i + 1}: Área no válida ({area}).")
    
def reevaluar_imagen(imagenes_no_detectadas):
    """
    Reevaluar las imágenes no detectadas previamente, usando parámetros ajustados.
    """
    for img_path in imagenes_no_detectadas:
        img = cv2.imread(img_path)  # Cargar la imagen
        if img is None:
            print(f"Error: No se pudo cargar la imagen desde '{img_path}'")
            continue
        
        # Mostrar la imagen original
        imshow(img, title="Imagen original", color_img=True)

        # Preprocesar la imagen
        img_close = preprocesar_imagen(img, threshold=63, canny_min=200, canny_max=350, kernel_size=(4, 3), iterations=2)

        # Filtrar componentes conectadas según el área
        filtered_img = filtrar_componentes_conectadas(img_close)

        # Refinar la imagen con una operación de apertura
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 4))
        img_open = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel, iterations=3)
        imshow(img_open, title="Refinamiento: Apertura")

        # Aplicar operación morfológica de cierre
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel, iterations=9)
        imshow(img_close, title="Refinamiento: Clausura")
             
        # Detectar posibles patentes en la imagen refinada
        posibles_patentes = encontrar_patentes_04(img_open, img)
        print(f"Se encontraron {len(posibles_patentes)} posibles patentes.")

        if len(posibles_patentes) == 0:
            print("No se encontraron posibles patentes.")
        else:    
          print(f"Se encontraron {len(posibles_patentes)} posibles patentes.")

        # Si se detectaron patentes, procesar la primera patente
        if posibles_patentes:
            primera_patente = posibles_patentes[0]  # Obtener la primera patente detectada
            imshow(primera_patente, title="Patente a color RGB")

            # Convertir la patente a escala de grises y mostrarla
            primera_patente_gray = cv2.cvtColor(primera_patente, cv2.COLOR_BGR2GRAY)
            imshow(primera_patente_gray, title="Patente recortada en blanco y negro")

            # Procesar las letras de la patente
            letras_recortadas = procesar_patente(primera_patente_gray)
            print(f"Se encontraron {len(letras_recortadas)} letras en la primera patente.")

            # Mostrar y dilatar las letras encontradas
            for i, (letra, area) in enumerate(letras_recortadas):
                print(f"Letra {i + 1}: Área = {area}")
                if 16 < area < 125:  # Validar el área de la letra
                    # Aplicar dilatación para mejorar la visibilidad de la letra
                    kernel = np.ones((2, 1), np.uint8)
                    letra_dilatada = cv2.dilate(letra, kernel, iterations=1)
                    imshow(letra_dilatada, title=f"Letra {i + 1}")
                else:
                    print(f"Letra {i + 1}: Área no válida ({area}).")

# Ejecutar la función principal
if __name__ == "__main__":
    main()
    if len(imagen_a_reevaluar) != 0:
      reevaluar_imagen(imagen_a_reevaluar)
      