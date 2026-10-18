import os
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from clases.detector_personas import DetectorPersonas
from clases.detector_grounding_dino import DetectorGroundingDINO
from clases.contador_personas import ContadorPersonas
from clases.procesador_videos import ProcesadorVideo

DATA_DIR = BASE_DIR / "data_videos"
CSV_DIR = BASE_DIR / "resultados" / "csv"
VIDEOS_DIR = BASE_DIR / "resultados" / "videos"


def obtener_videos():
    extensiones = {".mp4", ".avi", ".mov", ".mkv"}

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de videos: {DATA_DIR}"
        )

    return sorted(
        archivo for archivo in DATA_DIR.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in extensiones
    )


def ejecutar_detector(tipo_modelo="yolo", solo_un_video=True,
                      nombre_video="Video1.mp4"):
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    videos = obtener_videos()

    if solo_un_video:
        videos = [v for v in videos if v.name == nombre_video]

    if not videos:
        print("No se encontró ningún video para procesar.")
        return

    if tipo_modelo == "yolo":
        detector = DetectorPersonas()
        sufijo = "yolo"
    elif tipo_modelo == "grounding_dino":
        detector = DetectorGroundingDINO()
        sufijo = "grounding_dino"
    else:
        raise ValueError(
            "Modelo no válido. Usá 'yolo' o 'grounding_dino'."
        )

    for ruta_video in videos:
        nombre = ruta_video.stem
        ruta_salida_video = VIDEOS_DIR / f"{nombre}_{sufijo}.mp4"
        ruta_salida_csv = CSV_DIR / f"{nombre}_{sufijo}.csv"

        contador = ContadorPersonas()
        procesador = ProcesadorVideo(detector, contador)

        print(f"\nProcesando {ruta_video.name} con {sufijo}...")

        df = procesador.procesar(
            str(ruta_video),
            str(ruta_salida_video)
        )

        contador.guardar_csv(str(ruta_salida_csv))

        print("Fotogramas procesados:", len(df))

        if not df.empty:
            print(
                "Promedio de detecciones por fotograma:",
                round(df["cantidad_personas"].mean(), 2)
            )
            print(
                "Máximo de detecciones en un fotograma:",
                df["cantidad_personas"].max()
            )

        print("CSV:", ruta_salida_csv)
        print("Video:", ruta_salida_video)


def ejecutar_sam2(
    carpeta_frames,
    caja_inicial,
    nombre_salida="Video1_sam2_segmentado.mp4",
    fps_salida=12.5
):
    # SAM 2 se importa únicamente cuando se necesita.
    from clases.segmentador_sam2 import SegmentadorSAM2

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    segmentador = SegmentadorSAM2()
    ruta_salida = VIDEOS_DIR / nombre_salida

    cantidad = segmentador.segmentar_video(
        carpeta_frames=str(carpeta_frames),
        ruta_salida=str(ruta_salida),
        caja=np.asarray(caja_inicial, dtype=np.float32),
        fps_salida=fps_salida,
        obj_id=1
    )

    print("Fotogramas procesados por SAM 2:", cantidad)
    print("Video generado:", ruta_salida)


if __name__ == "__main__":
    print("Detector de personas")
    print("1. YOLOv8n")
    print("2. Grounding DINO")

    opcion = input("Seleccioná un modelo (1 o 2): ").strip()

    if opcion == "1":
        ejecutar_detector("yolo")
    elif opcion == "2":
        ejecutar_detector("grounding_dino")
    else:
        print("Opción no válida.")
