# split_dataset.py

import os
import shutil
import random
import argparse

def split_dataset(root_folder, output_folder, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1"

    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for species in os.listdir(root_folder):
        species_path = os.path.join(root_folder, species)
        if not os.path.isdir(species_path):
            continue

        os.makedirs(os.path.join(train_folder, species), exist_ok=True)
        os.makedirs(os.path.join(val_folder, species), exist_ok=True)
        os.makedirs(os.path.join(test_folder, species), exist_ok=True)

        audio_files = [f for f in os.listdir(species_path) if os.path.isfile(os.path.join(species_path, f))]
        random.shuffle(audio_files)

        total_files = len(audio_files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = total_files - train_count - val_count

        train_files = audio_files[:train_count]
        val_files = audio_files[train_count:train_count + val_count]
        test_files = audio_files[train_count + val_count:]

        if len(test_files) == 0:
            test_files = val_files
            val_files = []

        for file in train_files:
            shutil.copy(os.path.join(species_path, file), os.path.join(train_folder, species, file))
        for file in val_files:
            shutil.copy(os.path.join(species_path, file), os.path.join(val_folder, species, file))
        for file in test_files:
            shutil.copy(os.path.join(species_path, file), os.path.join(test_folder, species, file))

    print(f" División completada. Archivos copiados a '{output_folder}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide un conjunto de audios en train/val/test por especie.")
    parser.add_argument("--input", type=str, default="audios", help="Ruta al directorio raíz con carpetas por especie.")
    parser.add_argument("--output", type=str, default="TVT", help="Ruta del directorio de salida.")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Proporción de datos para entrenamiento.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proporción de datos para validación.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Proporción de datos para prueba.")
    args = parser.parse_args()

    split_dataset(args.input, args.output, args.train_ratio, args.val_ratio, args.test_ratio)

# ======================= EXECUTION========================
# DEFAULT EXECUTION 
# python Segment_Audio/split_folders.py

# default Args
# --input audios
# --output TVT
# --train_ratio 0.6
# --val_ratio 0.2
# --test_ratio 0.2