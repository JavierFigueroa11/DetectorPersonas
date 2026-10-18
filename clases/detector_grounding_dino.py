import torch
import cv2
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)


class DetectorGroundingDINO:

    def __init__(
        self,
        modelo_id="IDEA-Research/grounding-dino-tiny",
        confianza=0.35,
        confianza_texto=0.25
    ):
        self.dispositivo = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.confianza = confianza
        self.confianza_texto = confianza_texto

        self.procesador = AutoProcessor.from_pretrained(modelo_id)

        self.modelo = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(modelo_id)
            .to(self.dispositivo)
        )

        self.modelo.eval()

    def detectar(self, frame):
        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        entradas = self.procesador(
            images=imagen_rgb,
            text=[["person"]],
            return_tensors="pt"
        ).to(self.dispositivo)

        with torch.inference_mode():
            salidas = self.modelo(**entradas)

        alto, ancho = frame.shape[:2]

        resultados = (
            self.procesador.post_process_grounded_object_detection(
                salidas,
                entradas.input_ids,
                threshold=self.confianza,
                text_threshold=self.confianza_texto,
                target_sizes=[(alto, ancho)]
            )
        )

        return resultados[0]

    def anotar(self, frame, resultado):
        anotado = frame.copy()

        cajas = resultado["boxes"].cpu().tolist()
        confianzas = resultado["scores"].cpu().tolist()

        for caja, confianza in zip(cajas, confianzas):
            x1, y1, x2, y2 = map(int, caja)

            cv2.rectangle(
                anotado,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                anotado,
                f"person {confianza:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return anotado
