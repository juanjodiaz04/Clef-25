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

SR = 32000
TOP_K = 20  # Número de espectrogramas por página

# ========= Métricas =========
def calcular_psd(audio, sr):
    _, psd = welch(audio, sr, nperseg=1024)
    return np.var(psd)

def calcular_entropia_espectral(audio, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))**2
    suma = np.sum(S, axis=0, keepdims=True)
    suma[suma == 0] = 1e-10
    S_norm = S / suma
    entropias = entropy(S_norm, base=2, axis=0)
    return np.nanmean(entropias)

# ========= Evaluar y cargar todos los archivos =========
def cargar_segmentos(carpeta):
    resultados = []
    archivos = [f for f in os.listdir(carpeta) if f.endswith(".ogg")]

    psd_totales = []
    for fname in tqdm(archivos, desc="Cargando segmentos"):
        path = os.path.join(carpeta, fname)
        try:
            audio, sr = librosa.load(path, sr=SR)
            psd_var = calcular_psd(audio, sr)
            entropia = calcular_entropia_espectral(audio, sr)
            resultados.append((psd_var + entropia, path, audio))
            psd_totales.append(psd_var)
        except Exception as e:
            print(f"Error con {fname}: {e}")

    if psd_totales:
        print(f"\n Media de PSD en la carpeta '{carpeta}': {np.mean(psd_totales):.3e}")
    else:
        print(f"\n No se encontraron archivos válidos en la carpeta '{carpeta}'.")

    resultados.sort(reverse=True, key=lambda x: x[0])
    return resultados

# ========= Visualización e interacción =========
def mostrar_segmentos(segmentos):
    total = len(segmentos)
    pagina_actual = [0]
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.flatten()
    paths = []

    plt.subplots_adjust(bottom=0.2)
    boton_ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
    boton_ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
    boton_prev = Button(boton_ax_prev, 'Anterior')
    boton_next = Button(boton_ax_next, 'Siguiente')
    texto_ax = plt.axes([0.45, 0.05, 0.1, 0.075])
    texto_ax.axis("off")
    texto = texto_ax.text(0.5, 0.5, '', ha='center', va='center')

    def actualizar_pagina():
        paths.clear()
        for ax in axes:
            ax.clear()
            ax.axis('off')
        start = pagina_actual[0] * TOP_K
        end = min(start + TOP_K, total)
        subset = segmentos[start:end]

        for i, (score, path, audio) in enumerate(subset):
            S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            axes[i].imshow(S, aspect='auto', origin='lower', cmap='magma')
            axes[i].set_title(os.path.basename(path), fontsize=8)
            axes[i].axis('off')
            paths.append((audio, path))

        texto.set_text(f'Página {pagina_actual[0] + 1} / {((total - 1) // TOP_K) + 1}')
        fig.canvas.draw_idle()

    def onclick(event):
        for i, ax in enumerate(axes):
            if ax == event.inaxes and i < len(paths):
                audio, path = paths[i]
                print(f"Reproduciendo: {path}")
                sd.stop()
                sd.play(audio, SR)
                break

    def siguiente(event):
        if (pagina_actual[0] + 1) * TOP_K < total:
            pagina_actual[0] += 1
            actualizar_pagina()

    def anterior(event):
        if pagina_actual[0] > 0:
            pagina_actual[0] -= 1
            actualizar_pagina()

    boton_next.on_clicked(siguiente)
    boton_prev.on_clicked(anterior)
    fig.canvas.mpl_connect('button_press_event', onclick)

    actualizar_pagina()
    plt.show()

# ========= MAIN =========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizar y analizar archivos .ogg descartados")
    parser.add_argument("--species", type=str, required=True, help="Carpeta con archivos .ogg descartados")
    args = parser.parse_args()

    species_path = os.path.join("silence", args.species)
    segmentos = cargar_segmentos(species_path)
    if segmentos:
        mostrar_segmentos(segmentos)
    else:
        print("No hay segmentos para mostrar.")
