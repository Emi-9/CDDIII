import numpy as np
import statsmodels.api as sm
from analisis import AnalisisDescriptivo


class RegresionLineal:
    """
    Clase que permite el ajuste de un modelo de Regresion Lineal, que puede
    ser Regresion Lineal Simple y Regresion Lineal Multiple.
    Ambas clases depende de esta clase general.
    """
    def __init__(self, x, y):
        # x = variables predictora/s
        # y = variable respuesta
        self.x = x
        self.y = y
        self.resultado = None

    def ajustar_modelo(self):
        """
        Se ajuta el modelo de Regresión.
        """
        # se arma la matriz de diseño agregando la columna de unos
        X = sm.add_constant(self.x)
        # se estima el modelo de regresión lineal
        modelo = sm.OLS(self.y, X)
        self.resultado = modelo.fit()
        return self.resultado

    def parametros_modelo(self):
        """
        Retorna las estimaciones de los betas del modelo.
        """
        if self.resultado is None:
            print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
            parametros = self.resultado.params
            return parametros

    def ajustado_y(self):
        """Calula el valor predicho a partir del modelo ajustado
          de regresion lineal.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          y_pred = self.resultado.fittedvalues
          return y_pred

    def residuos(self):
        """ Calcula los residuos de entre los valores reales (y)
            y los valores dados por la recta d eminimos cuadrados (y_sombrero)
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          residuos = self.resultado.resid
          return residuos

    def estim_varianza_del_error(self):
        """Calcula la estimacion de la varianza del error
        """
        n = len(self.x)
        resid = self.residuos()
        var = np.sum( resid**2 ) / (n-2)
        return var

    def r_cuadrado(self):
        """ Calcula (R^2) el coeficiente de determinación y, es una medida de la
            proporción de la variabilidad que explica el modelo ajustado.
            valores de  R^2  cercanos a 1 son valores deseables para una buena
            calidad del ajuste.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          r_squared = self.resultado.rsquared
          return r_squared

    def r_ajustado(self):
        """Calcula el R² ajustado, es una corrección de  R^2  para permitir
          la comparación de modelos con distinta cantidad de regresoras.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          adjusted_r_squared = self.resultado.rsquared_adj
          return adjusted_r_squared

    def supuesto_normalidad(self):
      """Se verifica el supuesto de normalidad de los residuos, de manera
         grafica usando qqplot y de manera analítica usando shapiro test, usando
         el p-valor.
      """
      residuo = self.residuos()
      # grafica:
      rg = AnalisisDescriptivo(residuo)
      #miqqplot(residuo)
      rg.miqqplot()

      # test de normalidad:
      stat, p_valor1 = shapiro(residuo)
      print("\nValor p normalidad:", p_valor1)

    def supuestos_homocedasticidad(self):
      """Se verifica el supuesto de homocedasticidad de los residuos, de
          manera grafica y analítica por medio del p-valor.
      """
      # Homocedasticidad grafico
      predichos = self.ajustado_y()
      residuo = self.residuos()

      plt.scatter(predichos, residuo, marker="o", c='blue', s=30)
      plt.axhline(y=0, color='r', linestyle='--')  # Línea horizontal en y=0 para facilitar la visualización de los residuos
      plt.xlabel('Valores predichos')
      plt.ylabel('Residuos')
      plt.title('Gráfico de Residuos vs. Valores Predichos')
      plt.show()
      # Homocedasticidad test:
      X = sm.add_constant(self.x) # matriz de diseño
      bp_test = het_breuschpagan(residuo, X)# X es la matriz de diseño
      bp_value = bp_test[1]
      print("\nValor p Homocedasticidad:", bp_value)

    def int_confianza_betas(self, alfa):
        """funcion inicial: int_confianza_beta1(alfa, beta_1, var_estimada, t_crit)
        Calcula el intervalor de confianza para beta_1, a partir de un alfa (nivel
        de significacion) dado.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          IC = self.resultado.conf_int(alpha = alfa)
          print(f"Los Intervalos de confianza para los estimadores de beta son: {IC}")

    def p_valor_betas(self, b_i=0, i=1):
        """Es una funcion que retorna el p-valor de un test de hipotesis:
                          H_0: beta_i = k vs H_1 beta_i != k
            b_i: es el numero k sobre el cual se quiere hacer el test. Por
            default es 0.
            i: es el indice del beta que se quiere testear, es un natural i
            (i = 0, ..., n). Por default es 1.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          res = self.resultado
          SE_est = res.bse
          coef_xi = res.params[i]
          # valor de t observado:
          t_obs = (coef_xi - b_i)/SE_est[i]

          # el pvalor:
          X = res.model.exog # para recuperar la matriz de diseño del modelo
          grados_libertad = len(X[:, i]) - 2
          p_valor = 2 * stats.t.sf(abs(t_obs), df = grados_libertad)

          return p_valor

    def resumen_grafico(self, z):
        """ Grafico de dispersion de una variable cuantitativa predictora vs
            respuesta.
            z es la variable cuantitativa predictora que se quiere graficar.
        """
        plt.scatter(z, self.y, marker="o", c='blue', s=30)
        plt.xlabel('Variable Predictora')
        plt.ylabel('Variable Respuesta')
        plt.title('Gráfico de Dispersion: Var.Predict. vs. Var. Respuesta')
        plt.show()
