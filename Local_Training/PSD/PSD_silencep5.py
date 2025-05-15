#!/usr/bin/env python3
import os
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import welch
from scipy.stats import entropy
from matplotlib.widgets import Button
from tqdm import tqdm

# ========= Parámetros fijos =========
SR = 32000
TOP_K = 20
PERCENTILE = 100  # Percentil para determinar el umbral (default: 100%)

# ========= Métricas =========
def calcular_psd_ventanas(audio, sr, n_ventanas=20):
    """Calcula la varianza de la PSD en ventanas temporales."""
    duracion_total = len(audio) / sr
    duracion_ventana = duracion_total / n_ventanas
    samples_por_ventana = int(sr * duracion_ventana)

    psd_totales = []
    for i in range(n_ventanas):
        inicio = i * samples_por_ventana
        fin = inicio + samples_por_ventana
        segmento = audio[inicio:fin]
        if len(segmento) < 2:
            continue
        _, psd = welch(segmento, sr, nperseg=512)
        psd_totales.append(np.mean(psd))

    return np.var(psd_totales) if len(psd_totales) >= 2 else 0.0

def calcular_entropia_espectral_ventanas(audio, sr, n_ventanas=10, n_fft=2048, hop_length=512):
    """Calcula la varianza de la entropía espectral en ventanas temporales."""
    duracion_total = len(audio) / sr
    duracion_ventana = duracion_total / n_ventanas
    samples_por_ventana = int(sr * duracion_ventana)

    entropias = []
    for i in range(n_ventanas):
        inicio = i * samples_por_ventana
        fin = inicio + samples_por_ventana
        segmento = audio[inicio:fin]
        if len(segmento) < hop_length * 2:
            continue
        S = np.abs(librosa.stft(segmento, n_fft=n_fft, hop_length=hop_length))**2
        suma = np.sum(S, axis=0, keepdims=True)
        suma[suma == 0] = 1e-10
        S_norm = S / suma
        entropias.append(np.nanmean(entropy(S_norm, base=2, axis=0)))

    return np.var(entropias) if len(entropias) >= 2 else 0.0

# ========= Evaluar un archivo =========
def evaluar_segmento(path):
    """Carga un audio y calcula PSD y entropía en ventanas."""
    audio, _ = librosa.load(path, sr=SR)
    psd_var = calcular_psd_ventanas(audio, SR)
    entropia_var = calcular_entropia_espectral_ventanas(audio, SR)
    return psd_var, entropia_var, audio

# ========= Buscar y filtrar =========
def filtrar_segmentos(audio_folder, filt="psd"):
    """Filtra segmentos de audio basados en PSD y/o entropía espectral."""
    resultados = []
    valores_psd = []
    valores_entropia = []

    archivos = [f for f in os.listdir(audio_folder) if f.endswith(".ogg")]
    if not archivos:
        print(f"No se encontraron archivos .ogg en {audio_folder}")
        return []

    for fname in tqdm(archivos, desc=f"Procesando {audio_folder}"):
        path = os.path.join(audio_folder, fname)
        try:
            psd_var, ent_var, audio = evaluar_segmento(path)
            valores_psd.append(psd_var)
            valores_entropia.append(ent_var)
            resultados.append((psd_var, ent_var, path, audio))
        except Exception as e:
            print(f"Error con {fname}: {e}")

    if not resultados:
        return []

    psd_umbral = np.percentile(valores_psd, PERCENTILE)
    ent_umbral = np.percentile(valores_entropia, PERCENTILE)
    print(f"Umbral PSD (pct {PERCENTILE}%): {psd_umbral:.3e}")
    print(f"Umbral Entropy (pct {PERCENTILE}%): {ent_umbral:.3f}")

    descartados = []
    for psd_var, ent_var, path, audio in resultados:
        if filt == "psd" and psd_var < psd_umbral:
            score = psd_var
            descartados.append((score, path, audio))
        elif filt == "ent" and ent_var < ent_umbral:
            score = ent_var
            descartados.append((score, path, audio))

    descartados.sort(key=lambda x: x[0])
    print(f"Total descartados: {len(descartados)} ({len(descartados)/len(resultados)*100:.1f}%)")
    return descartados

# ========= Visualización interactiva =========
def mostrar_segmentos(descartados, filt="psd"):
    total = len(descartados)
    pagina = [0]
    fig, axes = plt.subplots(4, 5, figsize=(15,10))
    axes = axes.flatten()
    paths = []

    plt.subplots_adjust(bottom=0.2)
    ax_prev = plt.axes([0.3,0.05,0.1,0.075])
    ax_next = plt.axes([0.6,0.05,0.1,0.075])
    btn_prev = Button(ax_prev, "Anterior")
    btn_next = Button(ax_next, "Siguiente")
    txt_ax = plt.axes([0.45,0.05,0.1,0.075]); txt_ax.axis("off")
    txt = txt_ax.text(0.5,0.5,"", ha="center", va="center")

    def update():
        paths.clear()
        for ax in axes:
            ax.clear(); ax.axis("off")
        start = pagina[0]*TOP_K
        end = min(start+TOP_K, total)
        for i,(score,path,audio) in enumerate(descartados[start:end]):
            S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            axes[i].imshow(S, aspect="auto", origin="lower", cmap="magma")
            name = os.path.basename(path)
            label = f"{name} ({filt.upper()}: {score:.1e})"
            axes[i].set_title(label, fontsize=7)
            paths.append((audio,path))
        txt.set_text(f"Pág {pagina[0]+1}/{(total-1)//TOP_K+1}")
        fig.canvas.draw_idle()

    def on_click(event):
        for i,ax in enumerate(axes):
            if ax==event.inaxes and i<len(paths):
                audio,path = paths[i]
                print(f"Reproduciendo {os.path.basename(path)}")
                sd.stop(); sd.play(audio, SR)
                break

    def prev(event):
        if pagina[0]>0: pagina[0]-=1; update()
    def nxt(event):
        if (pagina[0]+1)*TOP_K<total: pagina[0]+=1; update()

    fig.canvas.mpl_connect("button_press_event", on_click)
    btn_prev.on_clicked(prev)
    btn_next.on_clicked(nxt)
    update()
    plt.show()

# ========= Main =========
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Filtrar y mostrar segmentos de audio")
    p.add_argument("--species", default="yelori1", help="Subcarpeta dentro de --input, o 'all'")
    p.add_argument("--input",   default="audios", help="Carpeta raíz de audios")
    p.add_argument("--filter",  default="psd", choices=["psd","ent"], help="Métrica de filtrado")
    args = p.parse_args()

    if args.species=="all":
        species_list = [d for d in os.listdir(args.input)
                        if os.path.isdir(os.path.join(args.input,d))]
    else:
        species_list = [args.species]

    for sp in species_list:
        folder = os.path.join(args.input, sp)
        print(f"\n=== {sp} ===")
        desc = filtrar_segmentos(folder, filt=args.filter)
        if desc:
            mostrar_segmentos(desc, filt=args.filter)
        else:
            print("No hay segmentos que mostrar.")


# ======================= EXECUTION========================
# EJEMPLO DE EJECUCIÓN:
# python PSD/PSD_silencep5.py --species yectyr1 --input audios --filter psd

# default Args
# --species yelori1
# --input audios
# --output testing
# --filter psd