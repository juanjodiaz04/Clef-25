#!/usr/bin/env python3
import os
import shutil
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Mover audios correspondientes a imágenes marcadas con '_voice.png'.")
    p.add_argument("--audio_dir", default="audios", help="Directorio raíz donde están los archivos .ogg, organizados por especie.")
    p.add_argument("--image_dir", required=True, help="Directorio raíz donde están las imágenes espectrograma organizadas por especie.")
    p.add_argument("--dest_dir", default="original/voice/", help="Directorio donde se moverán los audios con voz, manteniendo su subruta.")
    p.add_argument("--log_file", default="original/voice/voice_log.txt", help="Archivo .txt donde se escribirá el log de los movidos.")
    return p.parse_args()

def main():
    args = parse_args()

    moved = {}  # especie -> list of original relative paths

    # Walk through each species folder in image_dir
    for species in os.listdir(args.image_dir):
        img_species_dir = os.path.join(args.image_dir, species)
        if not os.path.isdir(img_species_dir):
            continue

        for fname in os.listdir(img_species_dir):
            if not fname.endswith("_voice.png"):
                continue

            # derive audio filename
            audio_fname = fname.replace("_voice.png", ".ogg")
            audio_path = os.path.join(args.audio_dir, species, audio_fname)
            if not os.path.isfile(audio_path):
                # no audio to move
                continue

            # prepare destination
            dest_species_dir = os.path.join(args.dest_dir, species)
            os.makedirs(dest_species_dir, exist_ok=True)
            dest_path = os.path.join(dest_species_dir, audio_fname)

            # move the file
            shutil.move(audio_path, dest_path)

            # record in log (use relative path with backslashes)
            rel_path = os.path.relpath(audio_path, start=args.audio_dir)
            rel_path = rel_path.replace(os.path.sep, "\\")
            moved.setdefault(species, []).append(rel_path)

    # write the log
    with open(args.log_file, "w", encoding="utf-8") as logf:
        for species in sorted(moved):
            paths = moved[species]
            logf.write(f"# {species} - {len(paths)} descartados\n")
            for p in paths:
                logf.write(f"audios\\{p}\n")
            logf.write("\n")

    print(f"Proceso completado. Log escrito en '{args.log_file}'.")

if __name__ == "__main__":
    main()
