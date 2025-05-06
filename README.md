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

## 2. Crear el espacio de trabajo

```bash
# Clonar el repositorio
git clone https://github.com/juanjodiaz04/Clef-25.git Workspace
cd Workspace

# Eliminar origen para evitar futuros pushes accidentales
git remote remove origin

cd Local_Training
# Crear carpetas necesarias
mkdir raw_audios
mkdir audios
mkdir embeddings
mkdir embeddings_csv
mkdir outputs
```

---

## 3. Configurar entornos virtuales (Python 3.10 recomendado)

```bash
# Verificar versión de Python
python --version

cd ~ Workspace

# Crear entorno virtual
py -3.10 -m venv env-class
py -3.10 -m venv env-emb

# Activar entorno virtual (Linux/Mac)
source env-class/bin/activate

# Activar entorno virtual (Windows)
source env-class/Scripts/activate
```

---

## 📦 4. Instalar requerimientos de BirdNET

```bash

cd ~ Workspace

(Embedder)
pip install -r Local_Training/BirdNET-Analyzer-1.5.1/requirements.txt

(Classifier)
pip install -r Req_classifier.txt
```

---

## 🎼 5. Obtener los Embeddings

```bash
cd BirdNET-Analyzer-1.5.1

# Ejecutar generación de embeddings
python -m birdnet_analyzer.embeddings --i ../audios/ --o ../embeddings/ --threads 4
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
