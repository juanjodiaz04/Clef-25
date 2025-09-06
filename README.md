# 🐦 BirdCLEF Pipeline – Workstation Setup  

This repository contains the code for the submission to **NeurIPS (BirdCLEF 2025 challenge)**.  
It provides the complete pipeline for training and inference on a local workstation.  
The workflow integrates **BirdNET Analyzer v2.4**, included within this repository, originally developed by the [BirdNET team](https://github.com/birdnet-team/BirdNET-Analyzer), to generate embeddings used by the classifiers.  


---

## 🔧 1. Verify and Install `ffmpeg`

```bash
# Check if ffmpeg is installed
ffmpeg -version

# If not installed
sudo apt update
sudo apt install ffmpeg
```

---

## 2. Create the Workspace

Download this repository and save it into a folder (~/Workspace)
```bash
cd Local_Training
# Create required folders
mkdir raw_audios
mkdir audios
mkdir embeddings
mkdir embeddings_csv
mkdir outputs
```

---

## 3. Set up Virtual Environments (Python 3.10 recommended) (~/Workspace)

```bash
# Check Python version
python --version

cd ..

# Create virtual environments
py -3.10 -m venv env-class
py -3.10 -m venv env-emb

# Activate virtual environment (Linux/Mac)
source env-class/bin/activate # Classification environment
source env-emb/bin/activate   # Embeddings environment

# Activate virtual environment (Windows)
source env-class/Scripts/activate

# Deactivate virtual environment
deactivate

```

---

## 4. Install Requirements for BirdNET and Classifier (~/Workspace)

```bash

# (Embedder)
source env-emb/bin/activate
pip install -r Local_Training/BirdNET-Analyzer-1.5.1/requirements.txt
deactivate

# (Classifier)
source env-class/bin/activate
pip install -r Req_classifier.txt
deactivate
```

---

## 5. Segment Audios (~/Local_Training)

```bash

# Activate classification environment from (~/Workspace)
source env-class/bin/activate

# Move to Local_Training folder
cd Local_Training

# Run segmentation of audios into 5s clips
python Segment_Audio/segment.py --threads 16

```

---

## 6. Identify Voices
```bash
# Identify voices with VGG19 model
python Audio_Processing/identify_voices.py --segmented_folder audios/ --output_root segmented_identified_images/

#Remove the identified voices from the segmented audios
python Audio_Processing/remove_voices.py audios/ path_to_image_folder --output_path segmented_identified_images/
```

---

## 7. Generate Embeddings (~/BirdNET-Analyzer-1.5.1)

```bash

# Move to BirdNET-Analyzer folder
cd BirdNET-Analyzer-1.5.1

# Run embedding generation
python -m birdnet_analyzer.embeddings --i ../audios/ --o ../embeddings/ --threads 16
deactivate

# Embedding generation by folder
python -m birdnet_analyzer.embeddings --i ../TVT/train/ --o ../TVT/emb_train/ --threads 16
python -m birdnet_analyzer.embeddings --i ../TVT/val/ --o ../embeddings/emb_val/ --threads 16
python -m birdnet_analyzer.embeddings --i ../TVT/test/ --o ../embeddings/emb_test --threads 16


```

---

## 8. Convert Embeddings to CSV (~Local_Training)

```bash

# Move to Workspace
cd ../.. 

# Activate classification environment from (~/Workspace)
source env-class/bin/activate

# Move to Local_Training folder
cd Local_Training

# Non-overlapping version (independent 5s chunks)
python embed2csv/embed_MT_P_NOV.py --threads 16

# Overlapping version
python embed2csv/embed_MT_P_OV.py --threads 16 

# CSV by folder
python embed2csv/embed_MT_P_OV.py --input TVT/emb_train/ --output embeddings_csv/train.csv --threads 16
python embed2csv/embed_MT_P_OV.py --input TVT/emb_val/ --output embeddings_csv/val.csv --threads 16
python embed2csv/embed_MT_P_OV.py --input TVT/emb_test/ --output embeddings_csv/test.csv --threads 16 

```

---

## 9. Train a Model (~Local_Training)

```bash

python Train_Inference/train.py --epochs 40 --model_type mlp

# Training with 60/20/20 folders
python Train_Inference/train_TVT.py --epochs 40 --model_type mlp

```

## 10. Inferencia Local (~Local_Training)

```bash 

python Train_Inference/inf_5s.py \
    --csv embeddings_csv/embeddings_MT_overlap.csv \
    --modelo outputs/run_06_0028/modelo_efficientnet_b7.pt \
    --labels outputs/run_06_0028/label_encoder.pkl \
    --sample-sub CSV/sample_submission.csv \
    --output outputs/run_06_0028/submission.csv \
    --model_type efficientnet_b7
