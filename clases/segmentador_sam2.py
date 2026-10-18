
import os
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


class SegmentadorSAM2:

    def __init__(
        self,
        checkpoint="/content/sam2/checkpoints/sam2.1_hiera_small.pt",
        config="/content/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
    ):
        """
        Inicializa el modelo SAM 2 para segmentación y seguimiento de video.
        """
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"No se encontró el checkpoint de SAM 2: {checkpoint}"
            )

        if not os.path.isfile(config):
            raise FileNotFoundError(
                f"No se encontró la configuración de SAM 2: {config}"
            )

        self.dispositivo = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.predictor = build_sam2_video_predictor(
            config,
            checkpoint,
            device=self.dispositivo
        )

    def segmentar_video(
        self,
        carpeta_frames,
        ruta_salida,
        caja,
        fps_salida=12.5,
        obj_id=1
    ):
        """
        Segmenta y sigue una persona a partir de una caja inicial.

        Parametros:
            carpeta_frames: carpeta con fotogramas JPG ordenados.
            ruta_salida: ruta del video de salida.
            caja: coordenadas [x1, y1, x2, y2] de la persona.
            fps_salida: fotogramas por segundo del video generado.
            obj_id: identificador del objeto seguido.

        Retorna:
            Cantidad de fotogramas procesados.
        """

        if not os.path.isdir(carpeta_frames):
            raise FileNotFoundError(
                f"No existe la carpeta de fotogramas: {carpeta_frames}"
            )

        archivos = sorted(
            archivo
            for archivo in os.listdir(carpeta_frames)
            if archivo.lower().endswith((".jpg", ".jpeg"))
        )

        if not archivos:
            raise ValueError(
                "No se encontraron fotogramas JPG en la carpeta."
            )

        caja = np.asarray(caja, dtype=np.float32)

        if caja.shape != (4,):
            raise ValueError(
                "La caja debe contener [x1, y1, x2, y2]."
            )

        primer_frame = cv2.imread(
            os.path.join(carpeta_frames, archivos[0])
        )

        if primer_frame is None:
            raise ValueError("No se pudo leer el primer fotograma.")

        alto, ancho = primer_frame.shape[:2]

        os.makedirs(
            os.path.dirname(os.path.abspath(ruta_salida)),
            exist_ok=True
        )

        escritor = cv2.VideoWriter(
            ruta_salida,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_salida,
            (ancho, alto)
        )

        if not escritor.isOpened():
            raise RuntimeError(
                f"No se pudo crear el video: {ruta_salida}"
            )

        inference_state = None
