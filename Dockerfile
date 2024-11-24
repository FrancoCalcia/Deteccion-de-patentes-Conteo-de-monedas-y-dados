# Usar una imagen base ligera de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar las dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar las dependencias de Python del archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutar los scripts Python en orden
CMD ["sh", "-c", "python monedas_dados.py && python plate.py"]
