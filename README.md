# 🐦 BirdCLEF Pipeline - Workstation Setup

Este documento describe paso a paso cómo configurar y ejecutar todo el pipeline de entrenamiento e inferencia en la Workstation.

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

## 3. Configurar entornos virtuales (Python 3.10 recomendado) (~/Workspace)

```bash
# Verificar versión de Python
python --version

cd ..

# Crear entorno virtual
py -3.10 -m venv env-class
py -3.10 -m venv env-emb

# Activar entorno virtual (Linux/Mac)
source env-class/bin/activate # Entorno de clasificación
source env-emb/bin/activate   # Entorno de embeddings

# Activar entorno virtual (Windows)
source env-class/Scripts/activate

# Desactivar entorno virtual
deactivate

```

---

## 4. Instalar requerimientos de BirdNET y el Clasificador(~/Workspace)

```bash

(Embedder)
source env-emb/bin/activate
pip install -r Local_Training/BirdNET-Analyzer-1.5.1/requirements.txt
deactivate

(Classifier)
source env-class/bin/activate
pip install -r Req_classifier.txt
deactivate
```

---

## 5. Segmentar audios (~/Local_Training)

```bash

# Escoger el entorno de clasificación desde (~/Workspace)
source env-class/bin/activate

# Moverse a la carpeta de Local_Training
cd Local_Training

# Ejecutar Segmentación de Audios a 5s
python Segment_Audio/segment.py --threads 16

```

---

## 6. Obtener los Embeddings (~/BirdNET-Analyzer-1.5.1)

```bash

# Moverse a la carpeta de BirdNET-Analyzer
cd BirdNET-Analyzer-1.5.1

# Ejecutar generación de embeddings
python -m birdnet_analyzer.embeddings --i ../audios/ --o ../embeddings/ --threads 16
deactivate

# Ejecutar generación de embeddings por folder
python -m birdnet_analyzer.embeddings --i ../TVT/train/ --o ../TVT/emb_train/ --threads 16
python -m birdnet_analyzer.embeddings --i ../TVT/val/ --o ../embeddings/emb_val/ --threads 16
python -m birdnet_analyzer.embeddings --i ../TVT/test/ --o ../embeddings/emb_test --threads 16

```

---

## 6. Convertir embeddings a CSV (~Local_Training)

```bash

# Moverse al Workspace
cd ../.. 

# Cambiar al entorno de clasificación desde (~/Workspace)
source env-class/bin/activate

# Moverse a la carpeta de Local_Training
cd Local_Training

# Versión sin solapamiento (chunks independientes de 5s)
python embed2csv/embed_MT_P_NOV.py --threads 16

# Versión con solapamiento (overlapping chunks)
python embed2csv/embed_MT_P_OV.py --threads 16 

# CSV por folder
python embed2csv/embed_MT_P_OV.py --input TVT/emb_train/ --output embeddings_csv/train.csv --threads 16
python embed2csv/embed_MT_P_OV.py --input TVT/emb_val/ --output embeddings_csv/val.csv --threads 16
python embed2csv/embed_MT_P_OV.py --input TVT/emb_test/ --output embeddings_csv/test.csv --threads 16 

```

---

## 7. Entrenar un modelo (~Local_Training)

```bash

python Train_Inference/train.py --epochs 40 --model_type mlp

# Entrenamiento con folders
python Train_Inference/train_TVT.py --epochs 40 --model_type mlp

```

## 8. Inferencia Local (~Local_Training)

```bash 

python Train_Inference/inf_5s.py     --csv embeddings_csv/embeddings_MT_overlap.csv     --modelo outputs/run_06_0028/modelo_efficientnet_b7.pt     --labels outputs/run_06_0028/label_encoder.pkl      --sample-sub CSV/sample_submission.csv     --output outputs/run_06_0028/submission.csv --model_type efficientnet_b7
