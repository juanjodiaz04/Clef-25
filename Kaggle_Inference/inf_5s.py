import os
import argparse
import torch
from torch import nn
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import my_models

def main():
    parser = argparse.ArgumentParser(description="Inferencia para BirdCLEF")
    parser.add_argument("--csv", type=str, required=True, help="Archivo CSV con embeddings")
    parser.add_argument("--modelo", type=str, required=True, help="Archivo .pt con pesos del modelo")
    parser.add_argument("--labels", type=str, required=True, help="Archivo .pkl con el LabelEncoder")
    parser.add_argument("--sample-sub", type=str, required=True, help="Archivo sample_submission.csv con columnas esperadas")
    parser.add_argument("--output", type=str, default="submission.csv", help="Archivo CSV de salida")
    parser.add_argument("--model_type", type=str, default="resnet18", help="Tipo de modelo a usar: 'resnet18' o 'mlp'")
    args = parser.parse_args()

    print("Cargando sample_submission...")
    sample_df = pd.read_csv(args.sample_sub)
    expected_columns = list(sample_df.columns)
    expected_labels = expected_columns[1:]  # sin 'row_id'

    print("Cargando modelo y LabelEncoder...")
    le = joblib.load(args.labels)
    num_classes = len(le.classes_)

    if args.model_type == "mlp":
        input_size = 32 * 96  # Aplanar las dimensiones de entrada (1x32x96)
        model = my_models.get_MLP(num_classes=num_classes, 
                                  input_size=input_size, 
                                  hidden_sizes=[256, 128, 64], 
                                  activation_fn=nn.ReLU)
    else:
        model = my_models.get_model(num_classes=num_classes,
                                    model_name=args.model_type)

    model.load_state_dict(torch.load(args.modelo, map_location="cpu"))
    model.eval()

    print("Cargando embeddings...")
    df = pd.read_csv(args.csv)
    #Borrar------
    df = df.sample(frac=0.1, random_state=42) # borrar esta línea para usar todo el dataset
    #-----------
    embedding_cols = [col for col in df.columns if col.startswith("emb_")]
    if not embedding_cols:
        raise ValueError("No se encontraron columnas que empiecen por 'emb_'.")
    print(f"Se encontraron {len(embedding_cols)} columnas de embedding.")
    X = df[embedding_cols].values.astype(np.float32)
    X = X.reshape(-1, 1, 32, 96)

    row_ids = []
    print("Realizando inferencia...")
    all_probs = []
    with torch.no_grad():
        for i, emb in enumerate(tqdm(X, desc="Inferencia")):
            input_tensor = torch.tensor(emb).unsqueeze(0)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

            # === Generar row_id personalizado ===
            base_filename = os.path.splitext(os.path.basename(df["row_id"].iloc[i]))[0]  # quita .ogg
            chunk_idx = int(df["group"].iloc[i])  # número del chunk
            end_time = (chunk_idx + 1) * 5
            row_id = f"{base_filename}_{end_time}"
            row_ids.append(row_id)

    # Crear DataFrame de predicciones
    pred_df = pd.DataFrame(all_probs, columns=le.classes_)
    pred_df["row_id"] = row_ids

    # Asegurar que todas las columnas esperadas estén presentes
    missing_cols = [col for col in expected_labels if col not in pred_df.columns]
    if missing_cols:
        # Crear un DataFrame con las columnas faltantes, todas con 0.0
        missing_df = pd.DataFrame(0.0, index=pred_df.index, columns=missing_cols)
        # Concatenar horizontalmente
        pred_df = pd.concat([pred_df, missing_df], axis=1)

    # Asegurar el orden correcto de columnas
    pred_df = pred_df[["row_id"] + expected_labels]

    # Convertir floats a precisión deseada
    pred_df[expected_labels] = pred_df[expected_labels].astype(np.float64)

    pred_df.to_csv(args.output, index=False, float_format="%.12f")
    print(f"Submission guardado como: {args.output}")

if __name__ == "__main__":
    main()

# ======================= EJECUCIÓN ========================
# python Inference/inf_5s.py     --csv Inference/embeddings_inference.csv     --modelo Train/outputs/modelo_05_1925.pt     --labels Train/outputs/label_encoder_05_1925.pkl      --sample-sub Inference/sample_submission.csv     --output Inference/submission.csv --model_type mlp

