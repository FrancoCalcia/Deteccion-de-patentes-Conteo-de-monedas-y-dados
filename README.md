
# **Trabajo Práctico N°2 - Procesamiento de Imágenes**

Este repositorio contiene dos scripts principales desarrollados para resolver problemas relacionados con el procesamiento de imágenes. Ambos scripts están diseñados para ejecutarse de forma automatizada y mostrar los resultados de cada etapa de procesamiento.

## **Descripción del proyecto**

### **Problema 1: Detección y clasificación de monedas y dados**
El script **`monedas_dados.py`** realiza las siguientes tareas basadas en la imagen `monedas.jpg`:
- **Segmentación automática** de monedas y dados.
- **Clasificación de monedas** según su tamaño o valor, y conteo automático.
- **Detección del valor** en la cara superior de los dados y conteo de cada valor.

El programa informa y muestra las imágenes procesadas en cada etapa.

---

### **Problema 2: Detección de patentes**
El script **`plate.py`** trabaja con un conjunto de imágenes de vehículos (`img<id>.png`) y realiza:
- **Detección automática de la placa patente** en las imágenes y su segmentación.
- **Segmentación de los caracteres** de la placa patente.
- Resultados visuales y detalles en cada etapa del procesamiento.

---

## **Estructura del proyecto**
```plaintext
.
├── Dockerfile              # Archivo de configuración para contenedor Docker
├── monedas_dados.py        # Script para resolver el problema 1
├── plate.py                # Script para resolver el problema 2
├── requirements.txt        # Dependencias del proyecto
├── archivos/               # Carpeta con las imágenes (monedas.jpg, img<id>.png)
└── README.md               # Documentación del proyecto
```

---

## **Requisitos**

Antes de ejecutar los scripts, asegúrate de tener los siguientes elementos instalados en tu sistema:
- **Python 3.8 o superior**
- Librerías listadas en `requirements.txt` (ver Instrucciones de Configuración).

O utiliza Docker para ejecutar el proyecto en un entorno aislado.

---

## **Instrucciones de configuración**

### Opción 1: Configuración manual

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/usuario/trabajo-practico-2.git
   cd trabajo-practico-2
   ```

2. **Instala las dependencias**:
   Asegúrate de estar en un entorno virtual (opcional pero recomendado) y ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecución de los scripts**:
   - Para el problema 1 (monedas y dados):
     ```bash
     python monedas_dados.py
     ```
   - Para el problema 2 (detección de patentes):
     ```bash
     python plate.py
     ```

---

### Opción 2: Uso de Docker

1. **Construye la imagen Docker**:
   ```bash
   docker build -t trabajo-practico-2 .
   ```

2. **Ejecuta el contenedor**:
   - Para el problema 1:
     ```bash
     docker run -it --rm trabajo-practico-2 python monedas_dados.py
     ```
   - Para el problema 2:
     ```bash
     docker run -it --rm trabajo-practico-2 python plate.py
     ```

---

## **Resultados esperados**

### **Problema 1: Monedas y dados**
El script mostrará:
1. Las monedas y los dados segmentados.
2. Clasificación y conteo automático de las monedas.
3. Los valores de las caras superiores de los dados y su conteo.

### **Problema 2: Patentes**
El script mostrará:
1. El área de la placa patente detectada y segmentada.
2. Los caracteres de la patente extraídos y segmentados.

Cada etapa del procesamiento genera imágenes de salida para visualización.

