
import pandas as pd


class ContadorPersonas:

    def __init__(self):
        self.registros = []

    def registrar(self, numero_frame, fps, cantidad_personas):
        #tiempo = round(numero_frame / fps, 2) -> Se cambia ya que el contador comienza en el fotograma 1.
        #Osea un tiempo distinto de cero.
        tiempo = round((numero_frame - 1) / fps, 2)


        self.registros.append({
            "numero_frame": numero_frame,
            "tiempo_segundos": tiempo,
            "cantidad_personas": cantidad_personas
        })

    def obtener_dataframe(self):
        columnas = [
            "numero_frame",
            "tiempo_segundos",
            "cantidad_personas"
        ]

        return pd.DataFrame(
            self.registros,
            columns=columnas
        )

    def guardar_csv(self, ruta_csv):
        df = self.obtener_dataframe()

        df.to_csv(
            ruta_csv,
            index=False,
            encoding="utf-8-sig"
        )

        return df
