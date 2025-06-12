import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from typing import Any
from analisis import AnalisisDescriptivo

# TODO: buscar una mejor manera de testear si se ajusto el modelo (DRY).

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
        Ajusta el modelo.
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
        if self.resultado is not None:
            return self.resultado.params
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def val_ajustados(self) -> pd.Series | np.ndarray[Any, Any]:
        '''
        Retorna el valor predicho a partir del modelo ajustado.
        '''
        if self.resultado is not None:
            return self.resultado.fittedvalues
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def residuos(self) -> pd.Series | np.ndarray[Any, Any]:
        '''
        Retorna los residuos (y vs ŷ).
        '''
        if self.resultado is not None:
            return self.resultado.resid
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def estim_var_error(self) -> np.float64:
        '''
        Retorna estimacion de la varianza del error.
        '''
        if self.resultado is not None:
            return self.resultado.mse_resid
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def r_cuadrado(self) -> np.float64:
        '''
        Retorna R² comun:
        * (valores cercanos a 1 son deseables)
        '''
        if self.resultado is not None:
            return self.resultado.rsquared
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def r_ajustado(self) -> np.float64:
        '''
        Retorna R² ajustado:
        * (permite comparar modelos con distinta cantidad de regresores).
        '''
        if self.resultado is not None:
            return self.resultado.rsquared_adj
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def supuesto_normalidad(self) -> None:
        '''
        Se verifica el supuesto de normalidad de los residuos, de manera
        grafica usando qqplot, de manera analitica usando el test de shapiro
        y el p-valor.
        '''
        if self.resultado is not None:
            residuo = self.residuos()
            # Grafico:
            rg = AnalisisDescriptivo(residuo)
            rg.QQplot()

            # Normalidad:
            stat, p_valor1 = shapiro(residuo)
            print("\np-valor normalidad:", p_valor1)
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def supuesto_homocedasticidad(self) -> None:
        '''
        Se verifica el supuesto de homocedasticidad de los residuos, de
        manera grafica y analitica por medio del p-valor.
        '''
        if self.resultado is not None:
            # Grafico:
            predichos = self.val_ajustados()
            residuo = self.residuos()

            plt.scatter(predichos, residuo, marker="o", c="blue", s=30)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.xlabel("Valores Predichos")
            plt.ylabel("Residuos")
            plt.title("Residuos vs Valores Predichos")
            plt.show()

            # Homocedasticidad:
            X = sm.add_constant(self.x)
            bp_test = het_breuschpagan(residuo, X)
            bp_value = bp_test[1]
            print("\np-valor homocedasticidad: ", bp_value)
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def int_confianza_betas(self, alfa: float) -> None:
        '''
        Retorna el intervalo de confianza para beta_1:
        - alfa => nivel de significancia.
        '''
        if self.resultado is not None:
            return self.resultado.conf_int(alpha=alfa)
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    def p_valor_betas(self, b_i: float = 0,
                      i: int = 1) -> np.ndarray[Any, Any] | None:
        '''
        Retorna el p-valor del test de hipotesis:
        * H_0: beta_i = k vs H_1: beta_i != k.
        - b_i => es el numero k sobre el cual se quiere hacer el test.
        - i => es el indice del beta que se quiere testear (es un entero).
        '''
        if self.resultado is not None:
            res = self.resultado
            SE_est = res.bse
            coef_xi = res.params[i]
            t_obs = (coef_xi - b_i) / SE_est[i]
            X = res.model.exog
            grados_libertad = len(X[:, i]) - 2
            return 2 * stats.t.sf(abs(t_obs), df=grados_libertad)
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANES")

    def resumen_grafico(self, z) -> None:
        '''
        Grafica scatter plot (predictora vs respuesta):
        - z => variable cuantitativa predictora.
        '''
        if self.resultado is not None:
            plt.scatter(z, self.y, marker="o", c="blue", s=30)
            plt.xlabel("Variable Predictora")
            plt.ylabel("Variable Respuesta")
            plt.title("Variable Predictora vs Variable Respuesta")
            plt.show()
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")
