import os
import shutil
import argparse
def borrar_audios_voces(path_audios, path_imagenes, output_path=None):
    # si la imagen termina en _voice, borra el audio
    #recorrer todas las imagenes
    for subfolder in os.listdir(path_imagenes):
        path_subfolder = os.path.join(path_imagenes, subfolder)
        if os.path.isdir(path_subfolder):
            for imagen in os.listdir(path_subfolder):
                #print(os.path.splitext(path_subfolder)[0])
                if imagen.endswith("_voice.png"):
                    #obtenemos el nombre del audio
                    nombre_audio = imagen.replace("_voice.png", ".ogg")
                    #comprobamos si existe el audio
                    path_audio = os.path.join(path_audios, os.path.basename(path_subfolder), nombre_audio)
                    print(path_audio)
                    if os.path.exists(path_audio):
                        print(f"Borrando {path_audio}")                        
                        os.remove(path_audio)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Remove audio files corresponding to voice images.")
    parser.add_argument("path_audios", type=str, help="Path to the folder containing audio files.")
    parser.add_argument("path_imagenes", type=str, help="Path to the folder containing image files.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output path (not used).")

    args = parser.parse_args()
    borrar_audios_voces(args.path_audios, args.path_imagenes, args.output_path)
# Ejemplo de uso:
# python remove_voices.py path_to_audio_folder path_to_image_folder --output_path optional_output
