import numpy as np
from .regresion_lineal import RegresionLineal

# TODO: terminar de adaptar a mi estilo
# ^ (usabilidad y nombres de funciones mas sencillos)


class RegresionLinealMultiple(RegresionLineal):
    """
    Primero se instancia (), luego:
    - y_predict_x_new =>
    - resumen_modelo =>
    """
    def __init__(self, x, y) -> None:
        super().__init__(x, y)

    def __main__(self):
        pass

    def y_predict_x_new(self, x_new):
        """
        Retorna el valor de un y_predicho del modelo de regresion
        a partir de un nuevo valor de x.
        x_new debe ser una LISTA con los valores qeu toma la variable
        explicativa para predecir.
        """
        if self.resultado is None:
            print("Falta ajustar el modelo")
            return None

        else:
            res = self.resultado
            # Se usa sm.add_constant para manejar el intercepto automáticamente
            X_new_const = sm.add_constant(np.array(x_new).reshape(1, -1))
            # Reshape para asegurar que sea 2D
            prediccion = res.predict(X_new_const)[0]
            return prediccion

    def resumen_modelo(self):
        """
        Imprime el summary() del modelo ajustado.
        """
        if self.resultado is None:
            print("Falta ajustar el modelo")

        else:
            res = self.resultado
            print(res.summary())
