#!/usr/bin/env python3
import os
import argparse
import numpy as np
import librosa
from scipy.stats import kurtosis
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SR = 32000
PCT = 5   # percentil para umbral de kurtosis
N_VENTANAS = 10

def calcular_kurtosis_tiempo_ventanas(audio, sr=SR, n_ventanas=N_VENTANAS):
    """Varianza de la kurtosis de amplitud en n_ventanas iguales."""
    L = len(audio)
    size = L // n_ventanas
    kurts = []
    for i in range(n_ventanas):
        seg = audio[i*size:(i+1)*size]
        if len(seg) < 2:
            continue
        seg_norm = seg / (np.max(np.abs(seg)) + 1e-10)
        kurts.append(kurtosis(seg_norm, fisher=False))
    return np.var(kurts) if len(kurts) >= 2 else 0.0

def procesar_especie(input_root, especie):
    """Recorre los .ogg de la especie y devuelve lista de descartados."""
    folder = os.path.join(input_root, especie)
    archivos = [f for f in os.listdir(folder) if f.endswith(".ogg")]
    kurt_vals = []
    info = []

    # Paso 1: medir kurtosis en cada archivo
    for fn in archivos:
        p = os.path.join(folder, fn)
        try:
            audio, sr = librosa.load(p, sr=SR)
            k = calcular_kurtosis_tiempo_ventanas(audio, sr)
            kurt_vals.append(k)
            info.append((p, k))
        except Exception as e:
            print(f"Error con {p}: {e}")

    if not info:
        return []

    # Paso 2: umbral al percentil PCT
    umbral = np.percentile(kurt_vals, PCT)

    # Paso 3: descartar los de kurtosis < umbral
    descartados = [p for p, k in info if k < umbral]
    return descartados

def main():
    p = argparse.ArgumentParser(description="Detectar segmentos planos (silencios) usando kurtosis")
    p.add_argument("--input",   default="audios", help="Directorio raíz con subcarpetas por especie")
    p.add_argument("--species", default="yelori1", help="Nombre de especie o 'all'")
    p.add_argument("--threads", type=int, default=None, help="Máximo de hilos (por defecto = CPUs)")
    p.add_argument("--output",  default="discarded.txt", help="Archivo de log de descartados")
    args = p.parse_args()

    # Lista de especies
    if args.species.lower() == "all":
        especies = [
            d for d in os.listdir(args.input)
            if os.path.isdir(os.path.join(args.input, d))
        ]
    else:
        especies = [args.species]

    max_workers = args.threads or os.cpu_count()

    # Procesar en paralelo
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(procesar_especie, args.input, sp): sp
            for sp in especies
        }
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Especies"):
            sp = futures[fut]
            results[sp] = fut.result()

    # Escribir el log
    with open(args.output, "w") as logf:
        for sp in sorted(results):
            desc = results[sp]
            logf.write(f"# {sp} - {len(desc)} descartados\n")
            for path in desc:
                logf.write(path + "\n")
            logf.write("\n")

    print(f"\nProceso terminado. Log guardado en '{args.output}'.")

if __name__ == "__main__":
    main()

# ========== EXECUTION ========== #
# python PSD/kurt_silence.py --input audios --species all --threads 4 --output discarded.txt

#default Args
# --input audios
# --species yelori1
# --threads 4
# --output discarded.txt
# --log discarded.txt