#!/usr/bin/env python3
import os
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import welch
from scipy.stats import entropy, kurtosis
from matplotlib.widgets import Button
from tqdm import tqdm

# ========= Parámetros fijos =========
SR = 32000
TOP_K = 20
PERCENTILE = 5  # Percentil para determinar el umbral (default: 100%)

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

def calcular_kurtosis_tiempo_ventanas(audio, sr, n_ventanas=10):
    duracion_total = len(audio) / sr
    duracion_ventana = duracion_total / n_ventanas
    muestras_por_ventana = int(sr * duracion_ventana)
    kurts = []
    for i in range(n_ventanas):
        seg = audio[i*muestras_por_ventana:(i+1)*muestras_por_ventana]
        if len(seg) < 2:
            continue
        # normalizar amplitudes en [-1,1]
        seg_norm = seg / (np.max(np.abs(seg)) + 1e-10)
        # curtosis en tiempo
        kurts.append(kurtosis(seg_norm, fisher=False))
    return np.var(kurts) if len(kurts) >= 2 else 0.0

# ========= Evaluar un archivo =========
def evaluar_segmento(path):
    audio, _ = librosa.load(path, sr=SR)
    return (
        calcular_psd_ventanas(audio, SR),
        calcular_entropia_espectral_ventanas(audio, SR),
        calcular_kurtosis_tiempo_ventanas(audio, SR),
        audio
    )

# ========= Buscar y filtrar =========
def filtrar_segmentos(audio_folder, filt="psd"):
    resultados, v_psd, v_ent, v_kurt = [], [], [], []
    archivos = [f for f in os.listdir(audio_folder) if f.endswith(".ogg")]
    if not archivos:
        print(f"No .ogg en {audio_folder}"); return []

    for f in tqdm(archivos, desc=f"Procesando {audio_folder}"):
        p = os.path.join(audio_folder,f)
        try:
            psd_var, ent_var, kurt_var, audio = evaluar_segmento(p)
            resultados.append((psd_var, ent_var, kurt_var, p, audio))
            v_psd.append(psd_var); v_ent.append(ent_var); v_kurt.append(kurt_var)
        except:
            continue

    # Umbrales
    umbral_psd  = np.percentile(v_psd,  PERCENTILE)
    umbral_ent  = np.percentile(v_ent,  PERCENTILE)
    umbral_kurt = np.percentile(v_kurt, PERCENTILE)
    print(f"Umbrales → PSD:{umbral_psd:.3e}, Ent:{umbral_ent:.3f}, Kurt:{umbral_kurt:.3f}")

    descartados = []
    for psd_var, ent_var, kurt_var, path, audio in resultados:
        if filt=="psd"  and psd_var  < umbral_psd:
            score = psd_var
        elif filt=="ent" and ent_var < umbral_ent:
            score = ent_var
        elif filt=="kurt" and kurt_var < umbral_kurt:
            score = kurt_var
        else:
            continue
        descartados.append((score, path, audio))

    descartados.sort(key=lambda x: x[0])
    print(f"Descartados: {len(descartados)}/{len(resultados)}")
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
    p.add_argument("--filter",  default="kurt", choices=["psd","ent","kurt"], help="Métrica de filtrado")
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
# python PSD/PSD_silencep5.py --species yectyr1 --input audios --filter kurt

# default Args
# --species yelori1
# --input audios
# --output testing
# --filter psd