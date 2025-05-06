import os
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import time

def load_embedding(path, agg='mean'):
    embeddings = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            emb = list(map(float, parts[2].split(",")))
            embeddings.append(emb)

    if not embeddings:
        raise ValueError(f"No se pudo extraer ningún embedding válido de: {path}")
    
    emb_array = np.array(embeddings)
    if agg == 'mean':
        aggregated = np.mean(emb_array, axis=0)
    elif agg == 'max':
        aggregated = np.max(emb_array, axis=0)
    else:
        raise ValueError(f"Agregación no soportada: {agg}")
    
    return list(map(str, aggregated))

def parse_filename(path):
    file = os.path.basename(path)
    label = os.path.basename(os.path.dirname(path))
    audio_id = file.split(".")[0].rsplit("_", 1)[0]
    chunk_index = int(file.split("_")[1].split(".")[0])
    return label, audio_id, chunk_index

def generar_csv_embeddings(input_dir, output_csv, chunk_size=3, num_threads=4, agg='mean'):
    max_threads = min(num_threads, os.cpu_count())
    all_txt_files = []

    for label_folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".birdnet.embeddings.txt"):
                all_txt_files.append(os.path.join(folder_path, file))

    audio_chunks = defaultdict(list)

    print(f"Cargando archivos en paralelo con {max_threads} hilos usando agregación: {agg}")
    start_time = time.time()

    def process_file(path):
        emb = load_embedding(path, agg)
        label, audio_id, chunk_idx = parse_filename(path)
        return (label, audio_id, chunk_idx, emb)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_file, path) for path in all_txt_files]
        for future in as_completed(futures):
            try:
                label, audio_id, chunk_idx, emb = future.result()
                audio_chunks[(label, audio_id)].append((chunk_idx, emb))
            except Exception as e:
                print(f"Error en procesamiento de archivo: {e}")

    print("Procesando agrupaciones...")
    rows = []

    for (label, audio_id), chunks in audio_chunks.items():
        chunks.sort(key=lambda x: x[0])
        embeddings = [e[1] for e in chunks]

        for i in range(len(embeddings)):
            group = embeddings[i:i+chunk_size]
            if len(group) < chunk_size:
                group += [group[-1]] * (chunk_size - len(group))
            if len(group) == chunk_size:
                concatenated = sum(group, [])
                row = [audio_id, str(i), label] + concatenated
                rows.append(row)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    num_features = len(rows[0]) - 3 if rows else chunk_size * 1024
    columns = ["row_id", "group", "label"] + [f"emb_{i}" for i in range(num_features)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"CSV guardado en {output_csv} con {len(rows)} filas.")
    print(f"Tiempo total: {elapsed:.2f} segundos.")

# ======================= EJECUCIÓN ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de CSV desde embeddings BirdNET")
    parser.add_argument("--input", type=str, default="embeddings", help="Directorio raíz de entrada con carpetas por clase")
    parser.add_argument("--output", type=str, default="embeddings_csv/embeddings_MT_overlap.csv", help="Ruta del archivo CSV de salida")
    parser.add_argument("--chunks", type=int, default=3, help="Número de embeddings a concatenar por fila")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos para procesamiento paralelo")
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "max"], help="Método de agregación para múltiples líneas en un archivo de embedding")

    args = parser.parse_args()

    generar_csv_embeddings(
        args.input,
        args.output,
        chunk_size=args.chunks,
        num_threads=args.threads,
        agg=args.agg
    )

# ======================= EJEMPLO DE EJECUCIÓN ========================
#python embed2csv/embed_MT_P_OV.py --input embeddings --output embeddings_csv/embeddings_MT_noverlap.csv --chunks 3 --threads 12 --agg mean