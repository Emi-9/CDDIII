import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from .regresion_lineal import RegresionLineal

# TODO: terminar de pasar a mi estilo


class RegresionLinealSimple(RegresionLineal):
    """
    Primero se instancia (), luego:
    - estimacion_betas =>
    - graf_scatter_recta =>
    - y_predict_x_new =>
    - t_obs_b1 =>
    - reg_rechazo_b1 =>
    - p_valor_beta =>
    - int_confianza_betas =>
    - int_prediccion_y =>
    """

    def __init__(self, x, y) -> None:
        super().__init__(x, y)

    def __main__(self):
        pass

    def estimacion_betas(self):
        """
        Retorna una tupla (b_0, b_1) de los estimadores de beta_0 y beta_1,
        usando minimos cuadrados.
        """
        x_media = np.mean(self.x)
        y_media = np.mean(self.y)
        numerador = np.sum((self.x - x_media) * (self.y - y_media))
        denominador = np.sum((self.x - x_media)**2)
        b_1 = numerador / denominador
        b_0 = y_media - b_1 * x_media
        return (b_0, b_1)

    def graf_scatter_recta(self):
        """
        Grafica los puntos de la variable predictora vs vaiable de respuesta.
        Ademas grafica la recta de minimos cuadrados.
        """
        # el regresor lineal:
        b_0, b_1 = self.estimacion_betas()
        y_pred = b_0 + b_1 * self.x
        # el grafico:
        plt.scatter(self.x, self.y, marker="o", c='blue', label='Datos', s=30)
        plt.plot(self.x, y_pred, linestyle='--', color='red', label='Recta estimada')
        plt.legend()
        plt.xlabel("variable predictora")
        plt.ylabel("variable respuesta")
        plt.title('')
        plt.show()

    def y_predict_x_new(self, x_new):
        """
        Retorna el valor de un y_predicho del modelo de regresion
        a partir de un nuevo valor de x: x_new.
        """
        if self.resultado is None:
            print("Falta ajustar el modelo")

        else:
            res = self.resultado
            # Crear la matriz de diseño con el nuevo punto de predicción
            X_new = sm.add_constant(np.array([[x_new]]))
            prediccion = res.predict(X_new)
            return prediccion

    def t_obs_b1(self, b1=0):
        """
        Funcion que calcula del t observado para determinar el sgte. test
        H_0: beta_1 = 0 vs H_1: beta_1 != 0
        Donde b es el valor que toma beta_1, por defecto es 0, porque se evalua
        el test de arriba.
        Pero b=1 si se evaluara el test: H_0: beta_1 = 1 vs H_1: beta_1 != 1.
        """
        if self.resultado is None:
            print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
            res = self.resultado
            coef_x = res.params[1]
            # error estándar estimado para el estimador de los betas
            SE_est = res.bse

            # valor de t observado:
            t_obs = (coef_x - b1) / SE_est[1]  # SE_est[1] es el error estandar
            # estimado para el estimador beta_1
            return t_obs

    def reg_rechazo_b1(self, alfa):
        """
        Funcion que muestra la region de rechazo para la hipotesis nula H_0
        a favor  de aceptar la hipotesis alternativa H_1.
        """
        # Completar
        # alfa = 0.05
        grados_libertad = len(self.x) - 2
        t_crit = stats.t.ppf(1 - (alfa/2), df=grados_libertad)
        print(f"(-inf, {-t_crit}) U ({t_crit}, inf)")

    def p_valor_beta(self, b1=0):
        """
        calcula el p-valor para evaluar el test de hipotesis:
        H_0: beta_1 = 0 vs H_1: beta_1 != 0
        """
        t_observado = self.t_obs_b1(b1)
        grados_libertad = len(self.x) - 2
        p_valor = 2 * stats.t.sf(abs(t_observado), df=grados_libertad)
        return p_valor

    def int_confianza_betas(self, alfa):
        """
        Funcion inicial: int_confianza_beta1(alfa, beta_1, var_estimada, t_crit)
        Calcula el intervalor de confianza para beta_1, a partir de un alfa (nivel
        de significacion) dado.
        """

        if self.resultado is None:
            print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
            IC = self.resultado.conf_int()
            print(f"Intervalo de confianza para beta_1 es: {IC[1]}")

    def int_prediccion_y(self, metodo, x_new, alfa):
        """
        Calcula el intervalo de predccion de una Y, a partir de una x_new,
        usando los metodos:
        - Metodo 1:  Construir un intervalo de confianza para el valor esperado de
                     Y para un valor particular de  X , por ejemplo  x0 :  E(Y|X=x0)
        - Metodo 2: Construir un intervalo de predicción de  Y  para un valor
                    particular de  X , por ejemplo  x0 :  Y0 .
                    se obtiene un intervalo de confianza/prediccion de nivel (1-alfa)
        """
        if self.resultado is None:
            print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
            res = self.resultado
            # Crear la matriz de diseño con el nuevo punto de predicción:
            X_new = sm.add_constant(np.array([[1, x_new]]))
            prediccion = res.get_prediction(X_new)

            if metodo == 1:
                return prediccion.conf_int(alpha=alfa, obs=False)

            elif metodo == 2:
                return prediccion.conf_int(obs=True, alpha=alfa)
