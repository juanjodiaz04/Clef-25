import os
import argparse
import numpy as np
import librosa
from scipy.signal import welch
from scipy.stats import entropy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

SR = 32000

def calcular_psd(audio, sr):
    _, psd = welch(audio, sr, nperseg=1024)
    return np.var(psd)

def calcular_psd_ventanas(audio, sr, n_ventanas=10):
    duracion_total = len(audio) / sr
    duracion_ventana = duracion_total / n_ventanas
    samples_por_ventana = int(sr * duracion_ventana)
    
    psd_totales = []
    
    for i in range(n_ventanas):
        inicio = i * samples_por_ventana
        fin = inicio + samples_por_ventana
        segmento = audio[inicio:fin]

        if len(segmento) < 2:
            continue  # saltar si el segmento es demasiado corto

        _, psd = welch(segmento, sr, nperseg=512)
        psd_total = np.sum(psd)  # o np.mean(psd), dependiendo del criterio
        psd_totales.append(psd_total)
    
    return np.var(psd_totales)

def calcular_entropia_espectral(audio, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))**2
    suma = np.sum(S, axis=0, keepdims=True)
    suma[suma == 0] = 1e-10
    S_norm = S / suma
    S_safe = np.where(S_norm == 0, 1e-10, S_norm)
    entropias = -np.sum(S_norm * np.log2(S_safe), axis=0)
    return np.nanmean(entropias)

def evaluar_segmento(path):
    audio, sr = librosa.load(path, sr=SR)
    psd_var = calcular_psd(audio, sr)
    entropia = calcular_entropia_espectral(audio, sr)
    return psd_var, entropia

def procesar_carpeta(carpeta):
    valores_psd = []
    valores_entropia = []
    rutas = []

    archivos = [f for f in os.listdir(carpeta) if f.endswith(".ogg")]
    for fname in archivos:
        path = os.path.join(carpeta, fname)
        try:
            psd_var, entropia = evaluar_segmento(path)
            valores_psd.append(psd_var)
            valores_entropia.append(entropia)
            rutas.append((path, psd_var, entropia))
        except Exception as e:
            print(f"Error con {path}: {e}")

    if not valores_psd:
        return (os.path.basename(carpeta), [])

    psd_umbral = np.percentile(valores_psd, 5)
    entropia_umbral = np.percentile(valores_entropia, 5)

    descartados = []
    for path, psd, ent in rutas:
        if psd < psd_umbral:  # or ent < entropia_umbral:
            descartados.append(path)

    return (os.path.basename(carpeta), descartados)

def procesar_directorio_raiz_mt(directorio_raiz, archivo_log="discarded.txt", max_workers=None):
    subcarpetas = [
        os.path.join(directorio_raiz, d)
        for d in sorted(os.listdir(directorio_raiz))
        if os.path.isdir(os.path.join(directorio_raiz, d)) and d[0].isalpha()
    ]

    resultados = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(procesar_carpeta, carpeta): carpeta for carpeta in subcarpetas}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando carpetas"):
            resultado = future.result()
            resultados.append(resultado)

    with open(archivo_log, "w") as log_file:
        for nombre_carpeta, descartados in sorted(resultados):
            log_file.write(f"# {nombre_carpeta} - {len(descartados)} descartados\n")
            for archivo in descartados:
                log_file.write(f"{archivo}\n")
            log_file.write("\n")

# ========== MAIN ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtrado de segmentos silenciosos o planos en múltiples carpetas")
    parser.add_argument("--input", type=str, default="audios", help="Ruta al directorio raíz con subcarpetas por especie")
    parser.add_argument("--output", type=str, default="PSD/discarded.txt", help="Archivo de log de salida")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos paralelos")

    args = parser.parse_args()
    procesar_directorio_raiz_mt(args.input, args.output, args.threads)
    print(f"\n Proceso terminado. Log guardado en '{args.output}'.")

# ======================= EXECUTION========================
# DEFAULT EXECUTION 
# python PSD/silence_detection.py --threads 16

# default Args
# --input audios
# --output PSD/discarded.txt
# --threads 4