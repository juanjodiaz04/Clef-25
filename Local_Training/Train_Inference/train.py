import os
import argparse
import datetime
import time
import platform
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import my_models

torch.manual_seed(42)
np.random.seed(42)

def configurar_logs(output_dir, timestamp):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'log_{timestamp}.txt')
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    return log, log_file

def log_hardware(log):
    log("\nResumen del hardware:")
    log(f"  Plataforma: {platform.system()} {platform.release()}")
    log(f"  Procesador: {platform.processor()}")
    log(f"  PyTorch version: {torch.__version__}")
    log(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log("")

def cargar_datos(path_csv):
    df = pd.read_csv(path_csv, dtype={"label": str})
    X = df[[f'emb_{i}' for i in range(3072)]].values.astype(np.float32)
    y = df["label"].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    X_train = X_train.reshape(-1, 1, 32, 96)
    X_val = X_val.reshape(-1, 1, 32, 96)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    return train_ds, val_ds, le, num_classes

def entrenar_modelo(model, train_dl, device, log, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)
        log(f"Epoca {epoch}: Loss = {avg_loss:.4f}")

def evaluar_modelo(model, val_dl, le, device, output_dir, log, timestamp, title=""):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)
    labels = list(range(len(le.classes_)))
    target_names = [str(c) for c in le.classes_]
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    log("\nReporte de resultados:")
    log(report)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.title(f"Matriz de Confusion {title}")
    plt.grid(False)
    cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}_{title}.png")
    plt.savefig(cm_path)
    plt.close()
    log(f"\nMatriz de confusion guardada como '{cm_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador CNN con embeddings BirdNET")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al archivo CSV de embeddings")
    parser.add_argument("--output", type=str, default="outputs", help="Directorio base de salida")
    parser.add_argument("--epochs", type=int, default=20, help="Número de epocas de entrenamiento")
    parser.add_argument("--model_type", type=str, default="resnet18", help="Tipo de modelo a usar: 'resnet18' o 'mlp'")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%d_%H%M")
    run_output_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    log, log_file = configurar_logs(run_output_dir, timestamp)
    log_hardware(log)

    train_ds, val_ds, le, num_classes = cargar_datos(args.csv)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    if args.model_type == "mlp":
        input_size = 32 * 96
        model = my_models.get_MLP(num_classes=num_classes, input_size=input_size, hidden_sizes=[256, 128, 64], activation_fn=nn.ReLU)
    else:
        model = my_models.get_model(num_classes=num_classes, model_name=args.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log(f"Modelo cargado y movido a {device}")
    start_time = time.time()
    entrenar_modelo(model, train_dl, device, log, epochs=args.epochs)
    elapsed_time = time.time() - start_time
    evaluar_modelo(model, val_dl, le, device, run_output_dir, log, timestamp, title=args.model_type)

    model_path = os.path.join(run_output_dir, f"modelo_{args.model_type}.pt")
    encoder_path = os.path.join(run_output_dir, "label_encoder.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(le, encoder_path)
    log(f"\nModelo guardado en '{model_path}'")
    log(f"\nLabel encoder guardado en '{encoder_path}'")
    log(f"Tiempo total de entrenamiento: {elapsed_time:.2f} segundos")


