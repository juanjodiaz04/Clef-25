import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

SR = 32000
IB_THRESHOLD = 1
FACTOR_RUIDO = 2

def obtener_bandas_inactivas(audio, sr, threshold_db=-40, n_fft=1024, n_bandas=16):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft)), ref=np.max)
    bins_por_banda = S.shape[0] // n_bandas
    bandas = []
    for i in range(n_bandas):
        inicio = i * bins_por_banda
        fin = (i + 1) * bins_por_banda if i < n_bandas - 1 else S.shape[0]
        if np.mean(S[inicio:fin, :]) < threshold_db:
            bandas.append((inicio, fin))
    return bandas, S.shape

def rellenar_bandas_inactivas(audio, sr, bandas_inactivas, shape_S, n_fft=1024, factor_ruido=2):
    S_complex = librosa.stft(audio, n_fft=n_fft)
    rms = max(np.sqrt(np.mean(audio**2)), 1e-8)
    for inicio, fin in bandas_inactivas:
        ruido_mag = np.random.normal(0, rms * factor_ruido, size=S_complex[inicio:fin, :].shape)
        fase = np.exp(1j * np.angle(S_complex[inicio:fin, :]))
        S_complex[inicio:fin, :] += ruido_mag * fase
    return librosa.istft(S_complex)

def procesar_especie(input_dir, backup_dir, especie):
    carpeta = os.path.join(input_dir, especie)
    modificados = []
    for fname in tqdm(os.listdir(carpeta), desc=f"Procesando {especie}"):
        if not fname.endswith(".ogg"):
            continue
        path = os.path.join(carpeta, fname)
        try:
            audio, sr = librosa.load(path, sr=SR)
            bandas, shape = obtener_bandas_inactivas(audio, sr)
            if len(bandas) > IB_THRESHOLD:
                # añadimos ruido
                audio_mod = rellenar_bandas_inactivas(audio, sr, bandas, shape, factor_ruido=FACTOR_RUIDO)
                # mover original a backup
                dest = os.path.join(backup_dir, especie)
                os.makedirs(dest, exist_ok=True)
                os.replace(path, os.path.join(dest, fname))
                # guardar ruido en origen
                sf.write(path, audio_mod, sr)
                modificados.append(path)
        except Exception as e:
            print(f"Error con {path}: {e}")
    return modificados

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="audios", help="Directorio raíz con subcarpetas por especie")
    parser.add_argument("--backup", default="original/Gaussian", help="Directorio para respaldar originales")
    parser.add_argument("--species", default="yelori1", help="Nombre de especie o 'all'")
    parser.add_argument("--log", default="original/Gaussian/noise_log.txt", help="Archivo de log")
    args = parser.parse_args()

    # Determinar lista de especies
    if args.species.lower() == "all":
        especies = [d for d in os.listdir(args.input)
                    if os.path.isdir(os.path.join(args.input, d))]
    else:
        especies = [args.species]

    with open(args.log, "w") as logf:
        for sp in sorted(especies):
            descartados = procesar_especie(args.input, args.backup, sp)
            # escribir header
            logf.write(f"# {sp} - {len(descartados)} modificados\n")
            # escribir rutas
            for ruta in descartados:
                logf.write(ruta + "\n")
            logf.write("\n")

    print(f" Proceso terminado. Log guardado en '{args.log}'.")

# ========== EXECUTION ========== #
# python Audio_Processing/add_gaussian.py --input audios --backup original/Gaussian --species all

# default Args
# --input audios
# --backup original/Gaussian
# --species yelori1
# --log original/Gaussian/noise_log.txt
