import os
import argparse
import soundfile as sf
import numpy as np
import logging
from glob import glob

logging.basicConfig(
    filename="segmentador.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def obtener_archivos_ogg(input_dir):
    return [os.path.join(input_dir, afile) for afile in sorted(os.listdir(input_dir)) if afile.endswith('.ogg')]

def load_audio(path) -> np.array:
    with sf.SoundFile(path) as f:
        audio = f.read()
        samplerate = f.samplerate
    return audio, samplerate

def segmentar_audios(input_dir, output_dir, segment_duration=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_paths = obtener_archivos_ogg(input_dir)
    total_segmentos = 0

    for audio_path in audio_paths:
        try:
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            print(f"Segmentando: {audio_path}")  # 🔍 Imprimir ruta

            data, samplerate = load_audio(audio_path)
            duration = len(data) / samplerate

            if duration > segment_duration:
                num_segments = int(duration // segment_duration) + 1
                for i in range(num_segments):
                    start_sample = int(i * segment_duration * samplerate)
                    end_sample = int(min((i + 1) * segment_duration * samplerate, len(data)))
                    segment_data = data[start_sample:end_sample]

                    if len(segment_data) < segment_duration * samplerate and len(segment_data) > 0:
                        num_repeats = int(np.ceil((segment_duration * samplerate) / len(segment_data)))
                        segment_data = np.tile(segment_data, num_repeats)[:int(segment_duration * samplerate)]
                        logging.info(f"Extendiendo audio: {audio_path}")

                    if len(segment_data) > 0:
                        segment_path = os.path.join(output_dir, f"{filename}_{i}.ogg")
                        if not os.path.exists(segment_path):
                            sf.write(segment_path, segment_data, samplerate)
                            total_segmentos += 1

            else:
                i = 0
                segment_data = data
                if len(segment_data) < segment_duration * samplerate and len(segment_data) > 0:
                    num_repeats = int(np.ceil((segment_duration * samplerate) / len(segment_data)))
                    segment_data = np.tile(segment_data, num_repeats)[:int(segment_duration * samplerate)]
                    logging.info(f"Extendiendo audio: {audio_path}")

                if len(segment_data) > 0:
                    segment_path = os.path.join(output_dir, f"{filename}_{i}.ogg")
                    if not os.path.exists(segment_path):
                        sf.write(segment_path, segment_data, samplerate)
                        total_segmentos += 1

        except Exception as e:
            logging.error(f"Error procesando {audio_path}: {e}")
            print(f"Error procesando {audio_path}: {e}")

    print(f"\n Total de segmentos generados: {total_segmentos}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentador de audios en clips de 3 segundos")
    parser.add_argument("--input", type=str, required=True, help="Directorio de entrada con archivos de audio")
    parser.add_argument("--output", type=str, required=True, help="Directorio de salida para segmentos")
    args = parser.parse_args()

    segmentar_audios(args.input, args.output)
