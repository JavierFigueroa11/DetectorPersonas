
import cv2


class ProcesadorVideo:

    def __init__(self, detector, contador):
        self.detector = detector
        self.contador = contador

    def procesar(self, ruta_entrada, ruta_salida):

        video = cv2.VideoCapture(ruta_entrada)

        if not video.isOpened():
            raise RuntimeError(
                f"No se puede abrir el video: {ruta_entrada}"
            )

        salida = None

        try:
            fps = video.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25.0

            ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            salida = cv2.VideoWriter(
                ruta_salida,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (ancho, alto)
            )

            if not salida.isOpened():
                raise RuntimeError(
                    f"No se pudo crear el video: {ruta_salida}"
                )

            numero_frame = 0

            while True:
                correcto, frame = video.read()

                if not correcto:
                    break

                numero_frame += 1

                # Llamar al método correcto del detector
                resultado = self.detector.detectar(frame)

                cantidad_personas = len(resultado.boxes)

                self.contador.registrar(
                    numero_frame,
                    fps,
                    cantidad_personas
                )

                frame_anotado = resultado.plot()

                cv2.putText(
                    frame_anotado,
                    f"Personas detectadas: {cantidad_personas}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                salida.write(frame_anotado)

        finally:
            video.release()

            if salida is not None:
                salida.release()

        return self.contador.obtener_dataframe()
