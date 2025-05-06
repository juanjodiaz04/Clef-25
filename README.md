# 🐦 BirdCLEF Pipeline - Workstation Setup

Este documento describe paso a paso cómo configurar y ejecutar todo el pipeline de entrenamiento e inferencia en tu estación de trabajo para la competencia BirdCLEF.

---

## 🔧 1. Verificar e Instalar `ffmpeg`

```bash
# Verificar si ffmpeg está instalado
ffmpeg -version

# Si no está instalado
sudo apt update
sudo apt install ffmpeg
```

---

## 📁 2. Crear el espacio de trabajo

```bash
# Clonar el repositorio
git clone https://github.com/juanjodiaz04/BN_R1.5.1.git Workspace
cd Workspace

# Eliminar origen para evitar futuros pushes accidentales
git remote remove origin

# Crear carpetas necesarias
mkdir audios
mkdir embeddings
mkdir embeddings_csv
mkdir outputs
```

---

## 📦 3. Mover o descomprimir archivos de audio

```bash
# Mover archivos si es necesario
mv origen/archivo.txt folder/destino

# Descomprimir archivos ZIP
unzip archivo.zip -d audios/

# Descomprimir archivos TAR
tar -xvzf archivo.tar.gz -C carpeta_destino/
```

---

## 🐍 4. Configurar entorno virtual (Python 3.11 recomendado)

```bash
# Verificar versión de Python
python --version

# Crear entorno virtual
py -3.11 -m venv Emb-env

# Activar entorno virtual (Linux/Mac)
source Emb-env/bin/activate

# Activar entorno virtual (Windows)
source Emb-env/Scripts/activate
```

---

## 📦 5. Instalar requerimientos de BirdNET

```bash
pip install -r BirdNET-Analyzer-1.5.1/requirements.txt
```

---

## 🎼 6. Obtener los Embeddings

```bash
cd BirdNET-Analyzer-1.5.1

# Ejecutar generación de embeddings
python3 -m birdnet_analyzer.embeddings --i ../audios/ --o ../embeddings/ --t
```

---

## 🧮 7. Convertir embeddings a CSV

```bash
cd ../Workspace

# Versión sin solapamiento (chunks independientes de 3s)
python embed2csv/embed_MT_P_NOV.py --input embeddings --output embeddings_csv/sin_overlap.csv

# Versión con solapamiento (overlapping chunks)
python embed2csv/embed_MT_P_OV.py --input embeddings --output embeddings_csv/overlap.csv
```

---

## 🧠 8. Entrenar un modelo personalizado (opcional)

```bash
cd BirdNET-Analyzer-1.5.1
mkdir ../custom_model

# Ejecutar entrenamiento
python -m birdnet_analyzer.train --i ../audios/ --o ../custom_model/ --threads 4
```
