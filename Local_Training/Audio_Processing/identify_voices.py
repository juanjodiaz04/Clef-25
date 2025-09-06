import os
import shutil
import csv
import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display
import logging
import argparse

def load_model():
    try:
        model = models.vgg19(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load("vgg19_voice_classification.pth", weights_only=True))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        raise

def procesar_img_voz(audio_path, image_path, name):
    sr = 44100 # MODIFIED
    hop_length = 1024
    n_mels = 128

    model = load_model()  # Cargar el modelo en cada proceso

    try:
        y, sr_actual = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr_actual, hop_length=hop_length, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(5, 5))
        librosa.display.specshow(mel_spec_db, sr=sr_actual, hop_length=hop_length, cmap='magma')
        plt.axis("off")
        # MODIFIED
        
        #segment_filename = f'{image_path}/{(audio_path.split("/")[-1]).split(".")[0]}'
        segment_filename = os.path.join(image_path, name)
        plt.savefig(segment_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Clasificación con IA
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        image = Image.open(f"{segment_filename}.png").convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        try:
            with torch.no_grad():  # Desactiva gradientes para ahorrar memoria
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                prediction = "voice_img" if predicted.item() == 1 else "no_voice_img"
                logging.info(f"Predicción: {prediction} - Archivo: {segment_filename}.png")

            # Liberar memoria GPU
            del image, outputs, predicted
            torch.cuda.empty_cache()

            if prediction == "voice_img":
                os.rename(f"{segment_filename}.png", f"{segment_filename}_voice.png")

        except RuntimeError as e:
            logging.error(f"Error en clasificación de imagen: {e}")
            if "CUDA out of memory" in str(e):
                logging.error("CUDA sin memoria. Intentando liberar caché.")
                torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Error procesando {audio_path}: {e}")

    # Eliminar modelo para liberar RAM/GPU
    del model
    torch.cuda.empty_cache()

def procesar_archivos(segmented_folder, output_root="segmented_audio/"):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    #Recorre todas las subcarpetas de la carpeta segmented_folder
    for subfolder in os.listdir(segmented_folder):
        subfolder_path = os.path.join(segmented_folder, subfolder)
        #====BORRAR====#
        # si el subfolder empieza por un digito no lo procesa
        if subfolder[0].isdigit(): #or os.path.exists(os.path.join(output_root, subfolder)):
            print(f"subcarpeta {subfolder} es no ave") 
            continue
        for file in os.listdir(subfolder_path):
            if file.endswith(".ogg"):
                audio_path = os.path.join(subfolder_path, file)
                # Crear subcarpeta en output_root
                image_subfolder = os.path.join(output_root, subfolder)
                if not os.path.exists(image_subfolder):
                    os.makedirs(image_subfolder)
                #segment_filename = f"{audio_path.split('/')[-1].split('.')[0]}"
                procesar_img_voz(audio_path, image_subfolder, file.split('.')[0])

def main():
    parser = argparse.ArgumentParser(description="Procesamiento de archivos de audio para clasificación de voces")
    parser.add_argument("--segmented_folder", type=str, required=True, help="Carpeta con los archivos de audio segmentados")
    parser.add_argument("--output_root", type=str, default="segmented_audio/", help="Carpeta de salida para las imágenes procesadas")
    parser.add_argument("--model_path", type=str, default="vgg19_voice_classification.pth", help="Ruta al archivo del modelo preentrenado")
    parser.add_argument("--log_file", type=str, default="proceso_imag_seg.log", help="Archivo de log para registrar el proceso")
    args = parser.parse_args()

    # Configurar logging
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Usar los argumentos en el procesamiento
    procesar_archivos(args.segmented_folder, args.output_root)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()

# Ejemplo de uso:
# python identify_voices.py --segmented_folder path_to_segmented_folder --output_root path_to_output_folder --model_path path_to_model.pth --log_file path_to_log.log