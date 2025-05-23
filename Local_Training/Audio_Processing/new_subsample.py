import os
import random
import csv
import shutil
import numpy as np
def mediana_segmentos_subcarpeta(sub_carpeta):
    """
    Calcula la mediana de la cantidad de archivos en las subcarpetas de una carpeta dada.
    Args:
        carpeta (str): Ruta a la carpeta que contiene subcarpetas.
    Returns:
        float: Mediana de la cantidad de archivos en las subcarpetas.
    """
    cantidades = {}
    for file in os.listdir(sub_carpeta): # revisa todos los archivos en la carpeta
        unique_id = file.split("_")[0]
        cantidades[unique_id] = cantidades.get(unique_id, 0) + 1
    
    cantidades = list(cantidades.values())
    cantidades.sort()
    return np.median(np.array(cantidades))

def calcular_mediana_FULL_aves(root_folder):
    """
    Calcula la mediana de una subcarpeta
    Se asocia la cantidad de segmentos que empiezan por un mismo sufijo, y se saca la mediana
    """

    cantidades = []
    for subcarpeta in os.listdir(root_folder):
        ruta_subcarpeta = os.path.join(root_folder, subcarpeta)
        # las subcarpetas de aves empiezan con una letra
        if os.path.isdir(ruta_subcarpeta) and subcarpeta[0].isalpha(): 
            archivos = [f for f in os.listdir(ruta_subcarpeta) if os.path.isfile(os.path.join(ruta_subcarpeta, f))]
            cantidades.append(len(archivos))
    print(cantidades)
    cantidades.sort()
    n = len(cantidades)
    if n == 0:
        return 0
    elif n % 2 == 1:
        return cantidades[n // 2]
    else:
        return (cantidades[n // 2 - 1] + cantidades[n // 2]) / 2


def limitar_archivos_en_subcarpetas(carpeta_raiz, n_archivos_av, n_archivos_NO_av, carpeta_salida):
    """
    Recorre la carpeta raíz y selecciona aleatoriamente archivos en subcarpetas, 
    si es mayor a la mediana de la cantidad de archivos en las subcarpetas.

    Args:
        carpeta_raiz (str): Ruta a la carpeta raíz que contiene las subcarpetas.
        n_archivos_av (int): Límite máximo de archivos a tomar por subcarpeta.
        carpeta_salida (str): Ruta a la carpeta donde guardar los archivos seleccionados.
    """
    random.seed(42)  # Para reproducibilidad
    # Crear la carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Recorrer las subcarpetas en la carpeta raíz
    for subcarpeta in os.listdir(carpeta_raiz):
        ruta_subcarpeta = os.path.join(carpeta_raiz, subcarpeta)
        # Verificar si es una carpeta
        if os.path.isdir(ruta_subcarpeta):
            # Obtener todos los archivos en la subcarpeta
            # ===========================
            # AVEEEEEES
            # las subcarpetas de aves empiezan con una letra
            if subcarpeta[0].isalpha():
                
                archivos = [f for f in os.listdir(ruta_subcarpeta) if os.path.isfile(os.path.join(ruta_subcarpeta, f))]
                # Si hay más archivos que el límite, seleccionar aleatoriamente
                if len(archivos) >= n_archivos_av:
                    archivos_seleccionados = random.sample(archivos, 500)
                else:
                    archivos_seleccionados = archivos
                print(f"Subcarpeta: {subcarpeta}, Archivos seleccionados: {len(archivos_seleccionados)}")

                # Crear la carpeta de salida para la subcarpeta
                carpeta_salida_subcarpeta = os.path.join(carpeta_salida, subcarpeta)
                os.makedirs(carpeta_salida_subcarpeta, exist_ok=True)

                # Copiar los archivos seleccionados a la carpeta de salida
                for archivo in archivos_seleccionados:
                    shutil.copy(os.path.join(ruta_subcarpeta, archivo), carpeta_salida_subcarpeta)



            #===========================
            ## NOO AVEEEEEES
            else:
                archivos = [f for f in os.listdir(ruta_subcarpeta) if os.path.isfile(os.path.join(ruta_subcarpeta, f))]
                # Si hay más archivos que el límite, seleccionar aleatoriamente
                if len(archivos) >= n_archivos_NO_av:
                    archivos_seleccionados = random.sample(archivos, 50)
                else:
                    archivos_seleccionados = archivos
                print(f"Subcarpeta: {subcarpeta}, Archivos seleccionados: {len(archivos_seleccionados)}")

                # Crear la carpeta de salida para la subcarpeta
                carpeta_salida_subcarpeta = os.path.join(carpeta_salida, subcarpeta)
                os.makedirs(carpeta_salida_subcarpeta, exist_ok=True)

                # Copiar los archivos seleccionados a la carpeta de salida
                for archivo in archivos_seleccionados:
                    shutil.copy(os.path.join(ruta_subcarpeta, archivo), carpeta_salida_subcarpeta)


carpeta_raiz = R"C:\Base_User\UdeA\Electronica\Proyectos\BirdClef_Paper\Clef-25\Local_Training\TVT\train"
#n_archivos = calcular_mediana_FULL_aves(carpeta_raiz)
carpeta_salida = R"C:\Base_User\UdeA\Electronica\Proyectos\BirdClef_Paper\Clef-25\Local_Training\TVT\train_augm"

n_archivos_NO_av = 50
n_archivos_av = 500

limitar_archivos_en_subcarpetas(carpeta_raiz, n_archivos_av,n_archivos_NO_av, carpeta_salida)