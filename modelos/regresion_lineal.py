import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from typing import Any
from analisis import AnalisisDescriptivo

sm_res = sm.regression.linear_model.RegressionResultsWrapper
# Es lo que retorna un modelo ajustado de sm, ignorar.


class RegresionLineal:
    '''
    Primero se instancia (toma x: predictora e y: respuesta), luego:
    - ajustar_modelo =>
    - parametros_modelo =>
    - val_ajustados =>
    - residuos =>
    - estim_var_eror =>
    - r_cuadrado =>
    - r_ajustado =>
    - supuesto_normalidad =>
    - supuesto_homocedasticidad =>
    - int_confianza_betas =>
    - p_valor_betas =>
    - resumen_grafico =>
    '''

    def __init__(self, x: list[Any] | np.ndarray[Any, Any] | pd.DataFrame,
                 y: list[Any] | np.ndarray[Any, Any] | pd.DataFrame) -> None:
        self.x = x
        self.y = y
        self.resultado = None

    def __main__(self):
        pass

    def ajustar_modelo(self) -> sm_res:
        '''
        Ajuta el modelo.
        '''
        x = self.x if len(self.x) > 1 else [[1, self.x]]
        X = sm.add_constant(x, has_constant="skip")
        # En caso de que ya tenga las constantes, no le agrega
        modelo = sm.OLS(self.y, X).fit()
        self.resultado = modelo
        return modelo

    def parametros_modelo(self) -> pd.Series | np.ndarray[Any, Any]:
        '''
        Retorna los ß estimados.
        '''
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
            return np.array([])
        else:
            return self.resultado.params

    def val_ajustados(self) -> pd.Series | np.ndarray[Any, Any]:
        '''
        Retorna el valor predicho a partir del modelo ajustado.
        '''
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
            return np.array([])
        else:
            return self.resultado.fittedvalues

    def residuos(self) -> pd.Series | np.ndarray[Any, Any]:
        '''
        Retorna los residuos (y vs ŷ).
        '''
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
            return np.array([])
        else:
            return self.resultado.resid

    def estim_var_error(self) -> np.float64:
        '''
        Retorna la estimacion de la varianza del error.
        '''
        return self.resultado.mse_resid

    def r_cuadrado(self) -> np.float64 | None:
        '''
        Retorna R² el coeficiente de determinacion y, es una medida de la
        proporcion de la variabilidad que explica el modelo ajustado.
        valores de R² cercanos a 1 son valores deseables para una buena
        calidad del ajuste.
        '''
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
            return None
        else:
            return self.resultado.rsquared

    def r_ajustado(self) -> np.float64 | None:
        '''
        Calcula el R² ajustado, es una correccion de  R²  para permitir
        la comparacion de modelos con distinta cantidad de regresoras.
        '''
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
            return None
        else:
            return self.resultado.rsquared_adj

    def supuesto_normalidad(self) -> None:
        '''
        Se verifica el supuesto de normalidad de los residuos, de manera
        grafica usando qqplot y de manera analitica usando shapiro test, usando
        el p-valor.
        '''
        residuo = self.residuos()
        # Grafico:
        rg = AnalisisDescriptivo(residuo)
        rg.QQplot()

        # Normalidad:
        stat, p_valor1 = shapiro(residuo)
        print("\nValor p normalidad:", p_valor1)

    def supuesto_homocedasticidad(self) -> None:
        '''
        Se verifica el supuesto de homocedasticidad de los residuos, de
        manera grafica y analitica por medio del p-valor.
        '''
        # Grafico:
        predichos = self.val_ajustados()
        residuo = self.residuos()

        plt.scatter(predichos, residuo, marker="o", c="blue", s=30)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Valores predichos")
        plt.ylabel("Residuos")
        plt.title("Gráfico de Residuos vs. Valores Predichos")
        plt.show()

        # Homocedasticidad:
        X = sm.add_constant(self.x)
        bp_test = het_breuschpagan(residuo, X)
        bp_value = bp_test[1]
        print("\nValor p Homocedasticidad: ", bp_value)

    def int_confianza_betas(self, alfa: float) -> None:
        '''
        funcion inicial: int_confianza_beta1(alfa, beta_1,
        var_estimada, t_crit)
        Calcula el intervalor de confianza para beta_1, a partir de un alfa
        (nivel de significacion) dado.
        '''
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
        else:
            IC = self.resultado.conf_int(alpha=alfa)
            print("Los Intervalos de confianza para los"
                  f"estimadores de beta son: {IC}")

    def p_valor_betas(self, b_i: float = 0,
                      i: int = 1) -> np.ndarray[Any, Any] | None:
        '''
        Es una funcion que retorna el p-valor de un test de hipotesis:
        H_0: beta_i = k vs H_1 beta_i != k
        b_i: es el numero k sobre el cual se quiere hacer el test. Por
        default es 0.
        i: es el indice del beta que se quiere testear, es un natural i
        (i = 0, ..., n). Por default es 1.
        '''
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
            return None
        else:
            res = self.resultado
            SE_est = res.bse
            coef_xi = res.params[i]
            t_obs = (coef_xi - b_i) / SE_est[i]
            X = res.model.exog
            grados_libertad = len(X[:, i]) - 2
            return 2 * stats.t.sf(abs(t_obs), df=grados_libertad)

    def resumen_grafico(self, z) -> None:
        '''
        Grafico de dispersion de una variable cuantitativa predictora vs
        respuesta.
        z es la variable cuantitativa predictora que se quiere graficar.
        '''
        plt.scatter(z, self.y, marker="o", c="blue", s=30)
        plt.xlabel("Variable Predictora")
        plt.ylabel("Variable Respuesta")
        plt.title("Gráfico de Dispersion: Var.Predict. vs. Var. Respuesta")
        plt.show()
