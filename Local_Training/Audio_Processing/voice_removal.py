#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display
import shutil

# ========= Configuración global =========
SR = 44100
HOP_LENGTH = 1024
N_MELS = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Funciones =========

def load_model(model_path):
    """Carga el modelo VGG19 adaptado para 2 clases."""
    model = models.vgg19(weights=None)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, weight_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def procesar_img_voz(model, audio_path, image_dir, base_name):
    """Genera espectrograma, lo clasifica y renombra según predicción."""
    try:
        y, sr_actual = librosa.load(audio_path, sr=SR)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr_actual, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Guardar imagen temporal
        os.makedirs(image_dir, exist_ok=True)
        img_path = os.path.join(image_dir, base_name + ".png")
        plt.figure(figsize=(5,5))
        librosa.display.specshow(mel_spec_db,
                                 sr=sr_actual,
                                 hop_length=HOP_LENGTH,
                                 cmap="magma")
        plt.axis("off")
        plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        # Transformación y clasificación
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(inp)
            _, pred = torch.max(out,1)
            etiqueta = "voice_img" if pred.item()==1 else "no_voice_img"
            logging.info(f"{base_name}.png -> {etiqueta}")
        # renombrar si voz
        if etiqueta=="voice_img":
            new_path = os.path.join(image_dir, base_name + "_voice.png")
            os.replace(img_path, new_path)
        return

    except Exception as e:
        logging.error(f"Error procesando {audio_path}: {e}")

def procesar_archivos(segmented_folder, output_root, model):
    """Recorre subcarpetas y procesa cada .ogg."""
    for sub in os.listdir(segmented_folder):
        subp = os.path.join(segmented_folder, sub)
        # omitir carpetas que empiecen con dígito
        if not os.path.isdir(subp) or sub[0].isdigit():
            logging.info(f"Omitiendo carpeta {sub}")
            continue
        for f in os.listdir(subp):
            if f.endswith(".ogg"):
                audio_path = os.path.join(subp, f)
                image_dir = os.path.join(output_root, sub)
                base_name = os.path.splitext(f)[0]
                procesar_img_voz(model, audio_path, image_dir, base_name)

def mover_audios_voces(path_audios, path_imagenes, carpeta_destino="audios_con_voz"):
    for subfolder in os.listdir(path_imagenes):
        path_subfolder = os.path.join(path_imagenes, subfolder)
        if not os.path.isdir(path_subfolder):
            continue

        # Carpeta destino para esta especie
        destino_sub = os.path.join(carpeta_destino, subfolder)
        os.makedirs(destino_sub, exist_ok=True)

        for imagen in os.listdir(path_subfolder):
            if not imagen.endswith("_voice.png"):
                continue

            # Derivar nombre de audio
            nombre_audio = imagen.replace("_voice.png", ".ogg")
            audio_origen = os.path.join(path_audios, subfolder, nombre_audio)

            if os.path.exists(audio_origen):
                audio_destino = os.path.join(destino_sub, nombre_audio)
                print(f"Moviendo {audio_origen} → {audio_destino}")
                shutil.move(audio_origen, audio_destino)
            else:
                print(f"¡No existe el audio esperado! {audio_origen}")

# ========= Main =========

def main():
    parser = argparse.ArgumentParser(description="Procesamiento de segmentos de audio para clasificación de voz.")
    parser.add_argument("--segmented_folder", type=str, required=True, help="Carpeta con archivos de audio segmentados.")
    parser.add_argument("--output_root", type=str, default="espectrogram_audio", help="Directorio donde se guardan las imágenes.")
    parser.add_argument("--model_path", type=str, default="vgg19_voice_classification.pth", help="Ruta al checkpoint del modelo.")
    parser.add_argument("--log_file", type=str, default="proceso_imag_seg.log", help="Archivo de log para registrar el proceso.")
    args = parser.parse_args()

    # Configurar logging
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("=== Inicio de procesamiento ===")
    logging.info(f"Segmented folder: {args.segmented_folder}")
    logging.info(f"Output root:      {args.output_root}")
    logging.info(f"Model path:       {args.model_path}")

    model = load_model(args.model_path)
    procesar_archivos(args.segmented_folder, args.output_root, model)

    logging.info("=== Fin de procesamiento ===")

if __name__ == "__main__":
    main()

# ================= EXECUTION =================
# python Audio_Processing/voice_removal.py --segmented_folder audios/ 
 
# default Args
# --output_root espectrogram_audio/ 
# --model_path vgg19_voice_classification.pth
# --log_file proceso_imag_seg.log
