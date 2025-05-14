import os
import argparse
import soundfile as sf
import shutil

def mover_segmentos_cortos(root_dir, output_dir, min_duracion=5.0):
    total_movidos = 0
    total_archivos = 0

    os.makedirs(output_dir, exist_ok=True)

    for especie in os.listdir(root_dir):
        especie_path = os.path.join(root_dir, especie)
        if not os.path.isdir(especie_path):
            continue

        especie_output = os.path.join(output_dir, especie)
        os.makedirs(especie_output, exist_ok=True)

        for fname in os.listdir(especie_path):
            if not fname.endswith(".ogg"):
                continue

            total_archivos += 1
            input_path = os.path.join(especie_path, fname)
            try:
                duracion = sf.info(input_path).duration
                if duracion < min_duracion:
                    output_path = os.path.join(especie_output, fname)
                    shutil.move(input_path, output_path)
                    print(f" Movido: {input_path} → {output_path} (duración: {duracion:.2f}s)")
                    total_movidos += 1
            except Exception as e:
                print(f" Error con {input_path}: {e}")

    print(f"\n Proceso completado. Archivos movidos: {total_movidos} de {total_archivos} analizados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mover archivos de audio .ogg de corta duración a otra carpeta.")
    parser.add_argument("--input", type=str, default="audios", help="Directorio raíz con subcarpetas de especies.")
    parser.add_argument("--output", type=str, default="short", help="Carpeta de salida para los archivos movidos.")
    parser.add_argument("--min_duration", type=float, default=4.0, help="Duración mínima en segundos.")
    args = parser.parse_args()

    mover_segmentos_cortos(args.input, args.output, args.min_duration)

# ======================= EXECUTION========================
# DEFAULT EXECUTION 
# python Segment_Audio/short_segments.py

# default Args
# --input audios 
# --output short 
# --min_duration 4.0
