import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from typing import Any
from .regresion_lineal import RegresionLineal

# TODO: terminar de adaptar a mi estilo
# ^ (usabilidad y nombres de funciones mas sencillos)

# TODO: DRY => cuando una funcion funciona con listas, arrays y
#       dataframes, ponerlo en una variable. Ocupa mucho espacio.


class RegresionLinealSimple(RegresionLineal):
    """
    Primero se instancia (toma x: predictora, y: respuesta), luego:
    - calcular_betas =>
    - resumen_grafico_simple =>
    - predecir_y =>
    - estadistico_t_beta1 =>
    - region_rechazo_beta1 =>
    - p_valor_beta1 =>
    - intervalo_confianza_beta1 =>
    - intervalo_prediccion_y =>
    """

    def __init__(self, x: list[Any] | np.ndarray[Any, Any] | pd.DataFrame,
                 y: list[Any] | np.ndarray[Any, Any] | pd.DataFrame) -> None:
        super().__init__(x, y)

    def __main__(self):
        pass

    def calcular_betas(self) -> tuple[Any, Any]:
        """
        Retorna los estimadores de beta_0 y beta_1 usando minimos cuadrados.
        """
        x_media = np.array(np.mean(self.x))
        y_media = np.array(np.mean(self.y))
        numerador = np.sum((self.x - x_media) * (self.y - y_media))
        denominador = np.sum((self.x - x_media)**2)
        b_1 = numerador / denominador
        b_0 = y_media - b_1 * x_media
        return (b_0, b_1)

    def resumen_grafico_simple(self) -> None:
        """
        Grafica predictora vs respuesta y recta de minimos cuadrados.
        """
        b_0, b_1 = self.calcular_betas()
        y_pred = b_0 + b_1 * self.x

        # Grafico:
        plt.scatter(self.x, self.y, marker="o", c="blue", label="Datos", s=30)
        plt.plot(self.x, y_pred, linestyle="--", color="red",
                 label="Recta estimada")
        plt.legend()
        plt.xlabel("variable predictora")
        plt.ylabel("variable respuesta")
        plt.title("")
        plt.show()

    def predecir_y(self, x_new: Any) -> Any:  # TODO: anotaciones
        """
        Retorna ŷ a partir de un x_new.
        """
        if self.resultado is not None:
            res = self.resultado
            X_new = sm.add_constant(np.array([[x_new]]))
            return res.predict(X_new)
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def estadistico_t_beta1(self, b1: float = 0) -> Any:  # TODO: anotaciones
        """
        Retorna el estadistico t del test de hipotesis:
        * H_0: beta_1 = 0 vs H_1: beta_1 != 0
        - b1 => beta_1
        """
        if self.resultado is not None:
            res = self.resultado
            coef_x = res.params[1]
            SE_est = res.bse
            return (coef_x - b1) / SE_est[1]
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def region_rechazo_beta1(self, alfa: float) -> str:
        """
        Retorna la region de rechazo para la hipotesis nula.
        - alfa => nivel de significancia a testear.
        """
        # TODO: Completar.
        grados_libertad = len(self.x) - 2
        t_crit = stats.t.ppf(1 - (alfa/2), df=grados_libertad)
        return f"(-inf, {-t_crit}) U ({t_crit}, inf)"

    def p_valor_beta1(self, b1: float = 0) -> Any:
        """
        Retorna el p-valor para:
        * H_0: beta_1 = 0 vs H_1: beta_1 != 0
        - b1 => beta_1
        """
        t_observado = self.estadistico_t_beta1(b1)
        grados_libertad = len(self.x) - 2
        return 2 * stats.t.sf(abs(t_observado), df=grados_libertad)

    def intervalo_confianza_beta1(self, alfa) -> Any:  # TODO: anotaciones
        """
        Retorna el intervalo de confianza para beta_1.
        - alfa => nivel de significancia.
        """
        if self.resultado is not None:
            IC = self.resultado.conf_int()
            return IC[1]
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def intervalo_prediccion_y(self, metodo, x_new, alfa):  # TODO: anotaciones
        """
        Calcula el intervalo de predccion de una Y, a partir de una x_new,
        usando los metodos:
        - Metodo 1:  Construir un intervalo de confianza para el valor
                     esperado de Y para un valor particular de  X,
                     por ejemplo  x0:  E(Y|X=x0)
        - Metodo 2: Construir un intervalo de predicción de  Y  para un valor
                    particular de  X, por ejemplo  x0:  Y0 .
                    se obtiene un intervalo de confianza/prediccion de
                    nivel (1-alfa)
        """
        if self.resultado is not None:
            res = self.resultado
            X_new = sm.add_constant(np.array([[1, x_new]]))
            prediccion = res.get_prediction(X_new)

            if metodo == 1:
                return prediccion.conf_int(alpha=alfa, obs=False)
            elif metodo == 2:
                return prediccion.conf_int(obs=True, alpha=alfa)
            else:
                raise ValueError("Tipo de metodo incorrecto")
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")
