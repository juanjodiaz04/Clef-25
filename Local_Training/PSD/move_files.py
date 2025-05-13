import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def mover_archivo(path, carpeta_destino, raiz_comun):
    path = path.strip()
    if not path or path.startswith("#"):
        return None

    if not os.path.isfile(path):
        return f"No encontrado: {path}"

    try:
        # Estructura relativa desde la raíz común
        rel_path = os.path.relpath(path, raiz_comun)
        destino_final = os.path.join(carpeta_destino, rel_path)

        # Crear carpetas intermedias si no existen
        os.makedirs(os.path.dirname(destino_final), exist_ok=True)

        shutil.move(path, destino_final)
        return None
    except Exception as e:
        return f"Error moviendo {path}: {e}"

def mover_archivos_mt(txt_path, carpeta_destino="silencio", max_workers=8):
    os.makedirs(carpeta_destino, exist_ok=True)

    with open(txt_path, "r") as f:
        lineas = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Inferir raíz común para conservar la estructura relativa
    rutas_validas = [line for line in lineas if os.path.isfile(line)]
    raiz_comun = os.path.commonpath(rutas_validas) if rutas_validas else ""

    total = 0
    errores = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(mover_archivo, linea, carpeta_destino, raiz_comun): linea
            for linea in lineas
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Moviendo archivos"):
            error = future.result()
            if error:
                errores.append(error)
            else:
                total += 1

    print(f"\n {total} archivos movidos a '{carpeta_destino}' conservando estructura.")
    if errores:
        print(f"⚠️ {len(errores)} errores encontrados:")
        for err in errores[:5]:
            print("  ", err)
        if len(errores) > 5:
            print("  ...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mover archivos listados en un .txt a una carpeta destino conservando estructura.")
    parser.add_argument("--input", type=str, default="PSD/discarded.txt", help="Ruta al archivo .txt con rutas de archivos a mover.")
    parser.add_argument("--output", type=str, default="silence", help="Carpeta destino donde se moverán los archivos.")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos a usar para mover archivos.")
    args = parser.parse_args()

    mover_archivos_mt(txt_path=args.input, carpeta_destino=args.output, max_workers=args.threads)