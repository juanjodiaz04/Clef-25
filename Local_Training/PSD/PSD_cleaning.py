import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from scipy.signal import welch
from scipy.stats import entropy
from matplotlib.widgets import Button
from tqdm import tqdm


# ========= Parámetros =========
UMBRAL_PSD_VAR = 1e-18
UMBRAL_ENTROPIA = 1.549e-18
SR = 32000  # Asume frecuencia de muestreo fija
TOP_K = 20  # Cuántos espectrogramas mostrar
FOLDER = "audios/yeofly1"  # Cambia esto a tu carpeta

# ========= Métricas =========
def calcular_psd(audio, sr):
    _, psd = welch(audio, sr, nperseg=1024)
    return np.var(psd)

def calcular_entropia_espectral(audio, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))**2
    suma = np.sum(S, axis=0, keepdims=True)
    suma[suma == 0] = 1e-10  # para evitar división por cero
    S_norm = S / suma
    entropias = entropy(S_norm, base=2, axis=0)
    return np.nanmean(entropias)

# ========= Evaluar un archivo =========
def evaluar_segmento(path):
    audio, sr = librosa.load(path, sr=SR)
    psd_var = calcular_psd(audio, sr)
    entropia = calcular_entropia_espectral(audio, sr)
    return psd_var, entropia, audio

# ========= Buscar y filtrar =========
def filtrar_segmentos(audio_folder):
    descartados = []
    valores_psd = []
    valores_entropia = []

    archivos = [f for f in os.listdir(audio_folder) if f.endswith(".ogg")]

    for fname in tqdm(archivos, desc="Procesando segmentos"):
        path = os.path.join(audio_folder, fname)
        try:
            psd_var, entropia, audio = evaluar_segmento(path)
            valores_psd.append(psd_var)
            valores_entropia.append(entropia)

            # print(f"{fname} → PSD var: {psd_var:.3e}, Entropía: {entropia:.2f}")

            if psd_var < UMBRAL_PSD_VAR : # or entropia < UMBRAL_ENTROPIA:
                score = psd_var + entropia
                descartados.append((score, path, audio))
        except Exception as e:
            print(f"Error con {fname}: {e}")
    
    # Ordenar por score (más altos = más dudosos)
    descartados.sort(reverse=True, key=lambda x: x[0])
    print(f"Total segmentos descartados: {len(descartados)}")

    psd_umbral = np.percentile(valores_psd, 5)
    entropia_umbral = np.percentile(valores_entropia, 5)
    print(f"Umbral PSD auto: {psd_umbral:.3e}")
    print(f"Umbral entropía auto: {entropia_umbral:.2f}")

    return descartados

# ========= Visualización e interacción =========
def mostrar_segmentos(descartados):
    total = len(descartados)
    pagina_actual = [0]  # mutable para que lo capture el closure
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.flatten()
    paths = []

    # Crear espacio adicional para botones
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
        subset = descartados[start:end]

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

# ========= Ejecución principal =========
if __name__ == "__main__":
    top_descartados = filtrar_segmentos(FOLDER)
    if not top_descartados:
        print("No se encontraron segmentos descartados.")
    else:
        mostrar_segmentos(top_descartados)
