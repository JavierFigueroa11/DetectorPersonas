from ultralytics import YOLO


class DetectorPersonas:

    def __init__(self, modelo="yolov8n.pt", confianza=0.25):
        self.modelo = YOLO(modelo)
        self.confianza = confianza

    def detectar(self, frame):
        resultado = self.modelo.predict(
            frame,
            classes=[0],
            conf=self.confianza,
            verbose=False
        )[0]

        return resultado