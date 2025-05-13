import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd

# ============ PARÁMETROS ============
FOLDER = "audios/yercac1"  # cambia esta ruta
SR = 32000
TOP_K = 20
UMBRAL_SILENCIO = 95  # porcentaje máximo tolerado
TOP_DB = 40  # umbral de energía para definir qué es silencio


# ============ FUNCIONES DE EVALUACIÓN ============

def calcular_porcentaje_silencio(audio, sr, top_db=30):
    total_duracion = len(audio) / sr
    if total_duracion == 0:
        return 100.0
    intervals = librosa.effects.split(audio, top_db=top_db)
    duracion_util = sum((end - start) for start, end in intervals) / sr
    porcentaje = max(0.0, 1.0 - duracion_util / total_duracion)
    return porcentaje * 100


def calcular_potencia_total(audio):
    return np.mean(audio**2)


# ============ PROCESAMIENTO DE ARCHIVOS ============

def procesar_archivos(folder):
    descartados = []

    for fname in os.listdir(folder):
        if not fname.endswith(".ogg"):
            continue
        path = os.path.join(folder, fname)
        try:
            audio, sr = librosa.load(path, sr=SR)
            silencio_pct = calcular_porcentaje_silencio(audio, sr, top_db=TOP_DB)
            potencia = calcular_potencia_total(audio)
            if silencio_pct > UMBRAL_SILENCIO:
                descartados.append((potencia, path, audio, silencio_pct))
        except Exception as e:
            print(f"Error con {fname}: {e}")

    descartados.sort(reverse=True, key=lambda x: x[0])  # ordenar por mayor potencia
    return descartados[:TOP_K]


# ============ VISUALIZACIÓN INTERACTIVA ============

def mostrar_segmentos(descartados):
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.flatten()

    paths = []

    for i, (potencia, path, audio, silencio_pct) in enumerate(descartados):
        ax = axes[i]
        S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = ax.imshow(S, aspect='auto', origin='lower', cmap='magma')
        fname = os.path.basename(path)
        ax.set_title(f"{fname}\nPot: {potencia:.3e}, Sil: {silencio_pct:.1f}%", fontsize=7)
        ax.axis('off')
        paths.append((audio, path))

    def onclick(event):
        for i, ax in enumerate(axes):
            if ax == event.inaxes:
                audio, path = paths[i]
                print(f"▶️ Reproduciendo: {path}")
                sd.stop()
                sd.play(audio, SR)
                break

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()


# ============ MAIN ============
if __name__ == "__main__":
    top_descartados = procesar_archivos(FOLDER)
    if not top_descartados:
        print("No se encontraron segmentos descartados por silencio.")
    else:
        mostrar_segmentos(top_descartados)
