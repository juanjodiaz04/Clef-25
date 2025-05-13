# =======================
# Imports y configuración inicial
# =======================

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
import my_models

from torch import nn
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Inferencia para BirdCLEF")
    parser.add_argument("--csv", type=str, required=True, help="Archivo CSV con embeddings")
    parser.add_argument("--run", type=str, required=True, help="ID del run (ej. 05_1925)")
    parser.add_argument("--sample-sub", type=str, default="CSV/sample_submission.csv", help="Archivo sample_submission.csv con columnas esperadas")
    parser.add_argument("--model_type", type=str, default="resnet18", help="Tipo de modelo a usar: 'resnet18' o 'mlp'")
    args = parser.parse_args()

    # Construir rutas automáticas desde el ID del run
    run_dir = os.path.join("outputs", f"run_{args.run}")
    model_path = os.path.join(run_dir, f"modelo_{args.model_type}.pt")
    labels_path = os.path.join(run_dir, "label_encoder.pkl")
    output_path = os.path.join(run_dir, "sample_submission.csv")

    print("Cargando sample_submission...")
    sample_df = pd.read_csv(args.sample_sub)
    expected_columns = list(sample_df.columns)
    expected_labels = expected_columns[1:]  # sin 'row_id'

    print("Cargando modelo y LabelEncoder...")
    le = joblib.load(labels_path)
    num_classes = len(le.classes_)

    if args.model_type == "mlp":
        input_size = 32 * 96  # Aplanar las dimensiones de entrada (1x32x96)
        model = my_models.get_MLP(num_classes=num_classes,
                                  input_size=input_size,
                                  hidden_sizes=[256, 128, 64],
                                  activation_fn=nn.ReLU)
    else:
        model = my_models.get_model(num_classes=num_classes,
                                    model_name=args.model_type,
                                    pretrained=False)  # No descargar pesos

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Cargando embeddings...")
    df = pd.read_csv(args.csv)
    embedding_cols = [col for col in df.columns if col.startswith("emb_")]
    if not embedding_cols:
        raise ValueError("No se encontraron columnas que empiecen por 'emb_'.")
    print(f"Se encontraron {len(embedding_cols)} columnas de embedding.")
    X = df[embedding_cols].values.astype(np.float32).reshape(-1, 1, 32, 96)

    row_ids = []
    all_probs = []

    print("Realizando inferencia...")
    with torch.no_grad():
        for i, emb in enumerate(tqdm(X, desc="Inferencia")):
            input_tensor = torch.tensor(emb).unsqueeze(0)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

            base_filename = os.path.splitext(os.path.basename(df["row_id"].iloc[i]))[0]
            chunk_idx = int(df["group"].iloc[i])
            end_time = (chunk_idx + 1) * 5
            row_id = f"{base_filename}_{end_time}"
            row_ids.append(row_id)

    pred_df = pd.DataFrame(all_probs, columns=le.classes_)
    pred_df["row_id"] = row_ids

    # Asegurar columnas esperadas
    missing_cols = [col for col in expected_labels if col not in pred_df.columns]
    if missing_cols:
        missing_df = pd.DataFrame(0.0, index=pred_df.index, columns=missing_cols)
        pred_df = pd.concat([pred_df, missing_df], axis=1)

    pred_df = pred_df[["row_id"] + expected_labels]
    pred_df[expected_labels] = pred_df[expected_labels].astype(np.float64)

    pred_df.to_csv(output_path, index=False, float_format="%.12f")
    print(f"Submission guardado como: {output_path}")

if __name__ == "__main__":
    main()

# ======================= EXECUTION========================
# DEFAULT EXECUTION 
# python Train_Inference/inf_2.py --csv embeddings_csv/embeddings_MT_overlap.csv --run 12_1323 --model_type efficientnet_b7


# default Args
# --sample-sub CSV/sample_submission.csv
# --model_type resnet18