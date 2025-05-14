import os
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from scipy.signal import welch
from scipy.stats import entropy
from matplotlib.widgets import Button
from tqdm import tqdm

# ========= Parámetros fijos =========
SR = 32000
TOP_K = 20

# ========= Argumentos por consola =========
parser = argparse.ArgumentParser(description="Filtrar y visualizar segmentos de una especie específica.")
parser.add_argument("--species", type=str, default="yelori1", help="Nombre de la subcarpeta dentro de 'audios'")
args = parser.parse_args()

FOLDER = os.path.join("audios", args.species)

# ========= Métricas =========
def calcular_psd(audio, sr):
    _, psd = welch(audio, sr, nperseg=1024)
    return np.var(psd)

def contar_bandas_inactivas(audio, sr, threshold_db=-40, n_fft=1024):
    # Calcular espectrograma en dB
    S = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft)), ref=np.max)

    # Dividir el eje de frecuencias en 16 bandas uniformes
    n_bandas = 16
    bins_por_banda = S.shape[0] // n_bandas

    bandas_inactivas = 0
    for i in range(n_bandas):
        inicio = i * bins_por_banda
        fin = (i + 1) * bins_por_banda if i < n_bandas - 1 else S.shape[0]
        banda = S[inicio:fin, :]
        energia_promedio = np.mean(banda)
        if energia_promedio < threshold_db:
            bandas_inactivas += 1

    return bandas_inactivas

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
        psd_total = np.median(psd)  
        psd_totales.append(psd_total)
    
    return np.var(psd_totales)

def calcular_entropia_espectral(audio, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))**2
    suma = np.sum(S, axis=0, keepdims=True)
    suma[suma == 0] = 1e-10  # para evitar división por cero
    S_norm = S / suma
    entropias = entropy(S_norm, base=2, axis=0)
    return np.nanmean(entropias)

# ========= Evaluar un archivo =========
def evaluar_segmento(path, IB_UMBRAL=4):
    audio, sr = librosa.load(path, sr=SR)

    # Si hay bandas inactivas, agregar ruido
    bandas_inactivas = contar_bandas_inactivas(audio, sr)
    if bandas_inactivas > IB_UMBRAL:  # Cambia este umbral según sea necesario
        audio = agregar_ruido_gaussiano(audio, snr_db=35)
        audio = rellenar_silencios(audio, threshold=1e-6, ruido_std=1e-5)

    psd_var = calcular_psd(audio, sr)
    entropia = calcular_entropia_espectral(audio, sr)

    return psd_var, entropia, audio

# ========= Ruido Aditivo =========
def agregar_ruido_gaussiano(audio, snr_db=35):
    rms_signal = np.sqrt(np.mean(audio**2))
    if rms_signal < 1e-8:  # evita división por 0 en señales muy bajas
        rms_signal = 1e-8
    snr_linear = 10 ** (snr_db / 10)
    rms_noise = rms_signal / np.sqrt(snr_linear)
    ruido = np.random.normal(0, rms_noise, audio.shape)
    return audio + ruido

def rellenar_silencios(audio, threshold=1e-6, ruido_std=1e-5):
    audio = audio.copy()
    zonas_planas = np.abs(audio) < threshold
    audio[zonas_planas] += np.random.normal(0, ruido_std, size=np.sum(zonas_planas))
    return audio

# ========= Buscar y filtrar =========
def filtrar_segmentos(audio_folder):
    resultados = []
    valores_psd = []
    valores_entropia = []

    archivos = [f for f in os.listdir(audio_folder) if f.endswith(".ogg")]

    # Paso 1: calcular todos los valores
    for fname in tqdm(archivos, desc="Procesando segmentos"):
        path = os.path.join(audio_folder, fname)
        try:
            psd_var, entropia, audio = evaluar_segmento(path)
            valores_psd.append(psd_var)
            valores_entropia.append(entropia)
            resultados.append((psd_var, entropia, path, audio))
        except Exception as e:
            print(f"Error con {fname}: {e}")

    # Paso 2: calcular umbrales
    psd_umbral = np.percentile(valores_psd, 5)
    entropia_umbral = np.percentile(valores_entropia, 5)
    print(f"Umbral PSD auto (percentil 5%): {psd_umbral:.3e}")
    print(f"Umbral entropía auto (percentil 5%): {entropia_umbral:.2f}")

    # Paso 3: filtrar usando umbrales
    descartados = []
    for psd_var, entropia, path, audio in resultados:
        #psd_var < psd_umbral: or 
        #entropia < entropia_umbral:
        if psd_var < psd_umbral:
            #score = entropia
            score = psd_var
            # score = psd_var + entropia
            descartados.append((score, path, audio))

    # Ordenar por score (opcional)
    descartados.sort(reverse=True, key=lambda x: x[0])
    print(f"Total segmentos descartados: {len(descartados)}")
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
                audio_ps, sr = librosa.load(path, sr=SR)
                psd_var = calcular_psd_ventanas(audio_ps, sr)
                entropia = calcular_entropia_espectral(audio_ps, sr)

                print(f"Reproduciendo: {path}", f"PSD: {psd_var:.3e}", f"Entropía: {entropia:.2f}")
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
