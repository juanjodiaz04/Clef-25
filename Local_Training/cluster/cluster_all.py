import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import soundfile as sf
import sounddevice as sd
import os

# ===================== CONFIGURACIÓN ===================== #
CSV_PATH = "embeddings_csv/embeddings_all.csv"  # CSV con múltiples especies
AUDIO_BASE_PATH = "audios"                      # Carpeta raíz de especies
EPS = 2                                   # DBSCAN eps (ajústalo después del gráfico)
MIN_SAMPLES = 20                                 # DBSCAN min_samples (y para k-distancia)
RANDOM_STATE = 42                               # Para t-SNE
# ========================================================= #

# 1. Cargar CSV
df = pd.read_csv(CSV_PATH)
print(f"CSV cargado con {len(df)} filas.")
emb_cols = [col for col in df.columns if col.startswith("emb_")]
X = df[emb_cols].astype(float).values

# 2. Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Mostrar gráfico de k-distancia

neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES)
neighbors_fit = neighbors.fit(X_scaled)
distances, _ = neighbors_fit.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.title(f"k-distancia para estimar eps (k = {MIN_SAMPLES})")
plt.xlabel("Punto ordenado")
plt.ylabel(f"Distancia al vecino #{MIN_SAMPLES}")
plt.grid(True)
plt.tight_layout()
plt.show()


# 4. DBSCAN
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
labels = db.fit_predict(X_scaled)
df["cluster"] = labels

# 5. t-SNE
tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
X_2d = tsne.fit_transform(X_scaled)
df["tsne_0"] = X_2d[:, 0]
df["tsne_1"] = X_2d[:, 1]

# 6. Reproducción de audio
def play_audio(row_id, group, label):
    filename = f"{row_id}_{group}.ogg"
    path = os.path.join(AUDIO_BASE_PATH, label, filename)
    if os.path.exists(path):
        print(f"Reproduciendo: {path}")
        try:
            sd.stop()
            data, samplerate = sf.read(path)
            sd.play(data, samplerate)
        except Exception as e:
            print(f"Error al reproducir {filename}: {e}")
    else:
        print(f"Archivo no encontrado: {path}")

# 7. Visualización interactiva
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df["tsne_0"], df["tsne_1"], c=df["cluster"], cmap="tab10", s=20)
ax.set_title("Clustering DBSCAN (t-SNE) - Múltiples especies")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.grid(True)

annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    index = ind["ind"][0]
    row_id = df.iloc[index]["row_id"]
    group = df.iloc[index]["group"]
    label = df.iloc[index]["label"]
    cluster = df.iloc[index]["cluster"]
    pos = df.iloc[index][["tsne_0", "tsne_1"]]
    annot.xy = pos
    text = f"{label}/{row_id}_{group}.ogg\ncluster {cluster}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)

def on_click(event):
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            index = ind["ind"][0]
            row_id = df.iloc[index]["row_id"]
            group = df.iloc[index]["group"]
            label = df.iloc[index]["label"]
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
            play_audio(row_id, group, label)
        else:
            annot.set_visible(False)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()
