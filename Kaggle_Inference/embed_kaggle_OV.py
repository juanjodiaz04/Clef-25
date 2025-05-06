import os
import pandas as pd
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

def load_embedding(path):
    embeddings = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # ignorar líneas mal formateadas
            emb = list(map(float, parts[2].split(",")))
            embeddings.append(emb)
    if not embeddings:
        raise ValueError(f"No se pudo extraer ningún embedding válido de: {path}")
    emb_avg = np.mean(embeddings, axis=0)
    return list(map(str, emb_avg))  # convertir de vuelta a strings para CSV

def parse_filename(path):
    file = os.path.basename(path)
    name = file.split(".")[0]
    if "_" not in name:
        raise ValueError(f"Nombre de archivo no tiene '_' para extraer chunk: {file}")
    *base_parts, chunk_part = name.split("_")
    audio_id = "_".join(base_parts)
    chunk_index = int(chunk_part)
    return audio_id, chunk_index

def generar_csv_embeddings_test(input_dir, output_csv, chunk_size=3, num_threads=4):
    max_threads = min(num_threads, os.cpu_count())
    all_txt_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.endswith(".birdnet.embeddings.txt")]

    audio_chunks = defaultdict(list)

    print(f"Cargando archivos en paralelo con {max_threads} hilos...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_path = {executor.submit(load_embedding, path): path for path in all_txt_files}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                emb = future.result()
                audio_id, chunk_idx = parse_filename(path)
                audio_chunks[audio_id].append((chunk_idx, emb))
            except Exception as e:
                print(f"Error en {path}: {e}")

    print("Procesando agrupaciones...")
    rows = []

    for audio_id, chunks in audio_chunks.items():
        chunks.sort(key=lambda x: x[0])
        embeddings = [e[1] for e in chunks]

        for i in range(len(embeddings)):
            group = embeddings[i:i+chunk_size]
            if len(group) < chunk_size:
                group += [group[-1]] * (chunk_size - len(group))
            if len(group) == chunk_size:
                concatenated = sum(group, [])
                row = [audio_id, i] + concatenated
                rows.append(row)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    num_features = len(rows[0]) - 2 if rows else chunk_size * 1024
    columns = ["row_id", "group"] + [f"emb_{i}" for i in range(num_features)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"CSV guardado en {output_csv} con {len(rows)} filas.")
    print(f"Tiempo total: {elapsed:.2f} segundos.")

# ======================= EJECUCIÓN ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de CSV para embeddings de test (sin etiquetas)")
    parser.add_argument("--input", type=str, default="Folder", help="Directorio de entrada con los archivos embeddings")
    parser.add_argument("--output", type=str, default="embeddings_csv/embeddings_test.csv", help="Ruta del archivo CSV de salida")
    parser.add_argument("--chunks", type=int, default=3, help="Número de embeddings a concatenar por fila")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos para procesamiento paralelo")
    args = parser.parse_args()

    generar_csv_embeddings_test(args.input, args.output, chunk_size=args.chunks, num_threads=args.threads)
