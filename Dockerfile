# Usar una imagen base ligera de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos de tu proyecto al contenedor
COPY . /app

# Instalar las dependencias del archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Se ejectura primero el archivo de monedas y cuando finaliza corre el otro.
CMD ["sh", "-c", "python monedas_dados.py && python plate.py"]

