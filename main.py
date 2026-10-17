import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))

from clases.detector_personas import DetectorPersonas
from clases.contador_personas import ContadorPersonas
from clases.procesador_videos import ProcesadorVideo


DATA_DIR = BASE_DIR / "data_videos"
CSV_DIR = BASE_DIR / "resultados" / "csv"
VIDEOS_DIR = BASE_DIR / "resultados" / "videos"


def main():

    extensiones = {".mp4", ".avi", ".mov", ".mkv"}

    videos = [
        archivo
        for archivo in DATA_DIR.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in extensiones
    ]

    if not videos:
        print("No hay videos en la carpeta data_videos.")
        print("Primero debes cargar un video.")
        return

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    detector = DetectorPersonas()

    for ruta_video in videos:

        nombre = ruta_video.stem

        ruta_salida_video = (
            VIDEOS_DIR / f"{nombre}_procesado.mp4"
        )

        ruta_salida_csv = (
            CSV_DIR / f"{nombre}_registro_personas.csv"
        )

        contador = ContadorPersonas()

        procesador = ProcesadorVideo(
            detector,
            contador
        )

        print(f"\nProcesando video: {ruta_video.name}")

        df = procesador.procesar(
            str(ruta_video),
            str(ruta_salida_video)
        )

        contador.guardar_csv(str(ruta_salida_csv))

        print("Procesamiento finalizado.")
        print("Fotogramas procesados:", len(df))

        if not df.empty:
            print(
                "Total de detecciones por fotograma:",
                df["cantidad_personas"].sum()
            )

            print(
                "Máximo de personas en un fotograma:",
                df["cantidad_personas"].max()
            )

        print("CSV generado:", ruta_salida_csv)
        print("Video procesado:", ruta_salida_video)


if __name__ == "__main__":
    main()