
import os
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


class SegmentadorSAM2:

    def __init__(
        self,
        checkpoint="/content/sam2/checkpoints/sam2.1_hiera_small.pt",
        config_dir="/content/sam2/sam2/configs",
        config="sam2.1/sam2.1_hiera_s.yaml"
    ):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"No se encontró el checkpoint: {checkpoint}"
            )

        if not os.path.isfile(os.path.join(config_dir, config)):
            raise FileNotFoundError(
                f"No se encontró la configuración: "
                f"{os.path.join(config_dir, config)}"
            )

        self.dispositivo = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Hydra necesita una ruta de configuración relativa
        # a la raíz de configuración del paquete SAM 2.
        import hydra
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        with hydra.initialize_config_dir(
            config_dir=os.path.abspath(config_dir),
            version_base=None
        ):
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
        if not os.path.isdir(carpeta_frames):
            raise FileNotFoundError(
                f"No existe la carpeta de fotogramas: {carpeta_frames}"
            )

        archivos = sorted(
            archivo for archivo in os.listdir(carpeta_frames)
            if archivo.lower().endswith((".jpg", ".jpeg"))
        )

        if not archivos:
            raise ValueError("No se encontraron fotogramas JPG.")

        caja = np.asarray(caja, dtype=np.float32)

        if caja.shape != (4,):
            raise ValueError("La caja debe tener [x1, y1, x2, y2].")

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
        cantidad_procesada = 0

        try:
            inference_state = self.predictor.init_state(
                video_path=carpeta_frames
            )

            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=caja
            )

            with torch.inference_mode():
                for frame_idx, object_ids, masks in (
                    self.predictor.propagate_in_video(inference_state)
                ):
                    ruta_frame = os.path.join(
                        carpeta_frames, f"{frame_idx:05d}.jpg"
                    )
                    frame = cv2.imread(ruta_frame)

                    if frame is None:
                        continue

                    ids = (
                        object_ids.tolist()
                        if hasattr(object_ids, "tolist")
                        else list(object_ids)
                    )

                    if obj_id in ids:
                        posicion = ids.index(obj_id)
                        mascara = masks[posicion]

                        if mascara.ndim == 3:
                            mascara = mascara[0]

                        mascara = (
                            mascara.detach().float().cpu().numpy() > 0
                        )

                        frame[mascara] = (
                            0.5 * frame[mascara].astype(np.float32)
                            + 0.5 * np.array(
                                [0, 255, 0], dtype=np.float32
                            )
                        ).astype(np.uint8)

                        cv2.putText(
                            frame,
                            f"SAM 2 - Objeto {obj_id}",
                            (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                    escritor.write(frame)
                    cantidad_procesada += 1

        finally:
            escritor.release()

            if inference_state is not None:
                self.predictor.reset_state(inference_state)

        print(f"Fotogramas procesados: {cantidad_procesada}")
        print(f"Video generado: {ruta_salida}")

        return cantidad_procesada
