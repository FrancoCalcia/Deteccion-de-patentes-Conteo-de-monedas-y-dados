import cv2
import matplotlib.pyplot as plt
import numpy as np

# Funciones auxiliares
def mostrar_imagen(img, titulo, cmap=None):
    """Muestra una imagen con título."""
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=cmap)
    plt.title(titulo)
    plt.axis("off")
    plt.show()

def mostrar_imagenes(imagenes, titulos, cmap=None):
    """Muestra varias imágenes en una misma fila."""
    n = len(imagenes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, titulo in zip(axes, imagenes, titulos):
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Cargar y preprocesar la imagen
def cargar_imagen(ruta):
    imagen = cv2.imread(ruta)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(desenfoque, 80, 180)
    bordes_dilatados = cv2.dilate(bordes, None, iterations=15)
    return imagen, gris, bordes, bordes_dilatados

# Detectar contornos
def detectar_contornos(bordes_dilatados, imagen_original):
    contornos, _ = cv2.findContours(bordes_dilatados.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagen_contornos = imagen_original.copy()
    cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)
    return contornos, imagen_contornos

# Clasificar monedas y guardar índices de recortes descartados
def clasificar_monedas_y_descartar(contornos, bordes, imagen):
    recortes = []
    indices_descartados = []
    conteo_monedas = {"1 peso": 0, "50 centavos": 0, "10 centavos": 0}

    for i, contorno in enumerate(contornos):
        # Recortar el contorno
        x, y, w, h = cv2.boundingRect(contorno)
        recorte = bordes[y:y+h, x:x+w]
        recortes.append(recorte)

        # Detectar círculos (monedas)
        circles = cv2.HoughCircles(
            recorte,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=250,
            param1=100,
            param2=65,
            minRadius=30,
            maxRadius=recorte.shape[0] // 2
        )

        # Procesar círculos detectados
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            recorte_color = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)
            for (cx, cy, r) in circles:
                # Dibujar círculos en el recorte
                cv2.circle(recorte_color, (cx, cy), r, (0, 255, 0), 2)
                area = np.pi * r**2

                # Clasificar según el área
                if 65000 <= area <= 89000:
                    conteo_monedas["1 peso"] += 1
                elif area > 90000:
                    conteo_monedas["50 centavos"] += 1
                elif 50000 <= area <= 62000:
                    conteo_monedas["10 centavos"] += 1

            mostrar_imagen(cv2.cvtColor(recorte_color, cv2.COLOR_BGR2RGB), f"Recorte {i+1}: Círculo Detectado")
            print(f"Recorte {i+1}: Centro=({cx}, {cy}), Radio={r}, Área={area:.2f}")
        else:
            # Si no es moneda, se descarta para ser procesado como dado
            indices_descartados.append(i)
            print(f"Recorte {i+1}: No se detectaron círculos (posible dado).")

    # Imprimir el conteo final de monedas y calcular el valor total
    print("\nConteo Final de Monedas:")
    valor_total = 0  # Inicializar el valor total de las monedas
    for tipo, cantidad in conteo_monedas.items():
        valor = 0
        if tipo == "1 peso":
            valor = 1.0
        elif tipo == "50 centavos":
            valor = 0.5
        elif tipo == "10 centavos":
            valor = 0.1
        valor_total += cantidad * valor
        print(f"{tipo}: {cantidad}")

    print(f"\nLa suma total de las monedas es: ${valor_total:.2f} pesos")

    return recortes, conteo_monedas, indices_descartados


# Procesar dados (solo recortes descartados)
def procesar_dados(indices_descartados, recortes, kernel_apertura, kernel_cierre):
    resultados_dados = []
    for i in indices_descartados:
        recorte = recortes[i]
        recorte_morfo = cv2.morphologyEx(recorte, cv2.MORPH_OPEN, kernel_apertura)
        recorte_morfo = cv2.morphologyEx(recorte_morfo, cv2.MORPH_CLOSE, kernel_cierre)

        # Detectar puntos (caras de dados)
        circles = cv2.HoughCircles(
            recorte_morfo,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=80,
            param2=30,
            minRadius=5,
            maxRadius=50
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            recorte_color = cv2.cvtColor(recorte_morfo, cv2.COLOR_GRAY2BGR)
            for (x, y, r) in circles:
                cv2.circle(recorte_color, (x, y), r, (0, 255, 0), 2)
            resultados_dados.append((i, len(circles)))
            mostrar_imagen(cv2.cvtColor(recorte_color, cv2.COLOR_BGR2RGB), f"Dado {i+1}: {len(circles)} puntos")
            print(f"Dado {i+1}: La cara del dado tiene {len(circles)} puntos.")
        else:
            print(f"Dado {i+1}: No se detectaron puntos.")

    return resultados_dados

# --- Ejecución del Código ---
ruta_imagen = 'archivos/monedas.jpg'
imagen, gris, bordes, bordes_dilatados = cargar_imagen(ruta_imagen)

mostrar_imagenes(
    [cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB), gris, bordes],
    ["Imagen Original", "Escala de Grises", "Bordes Detectados (Canny)"],
    cmap='gray'
)

contornos, imagen_contornos = detectar_contornos(bordes_dilatados, imagen)
mostrar_imagen(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB), "Contornos Detectados")
print(f"Total de objetos detectados: {len(contornos)}")

recortes, conteo_monedas, indices_descartados = clasificar_monedas_y_descartar(contornos, bordes, imagen)

kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
resultados_dados = procesar_dados(indices_descartados, recortes, kernel_apertura, kernel_cierre)