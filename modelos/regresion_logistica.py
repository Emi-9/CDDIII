import numpy as np
import random
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import auc


class RegresionLogistica:
    """
    Clase que ajusta un modelo de Regresion Logistica.
    Requerimiento: Las variables categoricas sean codificadas antes de ajustar
    el modelo.
    Data es una base de datos con variables que sean cuantitativas.
    """

    def __init__(self, data) -> None:
        self.data = data
        self.data_train = None
        self.data_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.resultado = None

    def separar_data_train_test(self, seed=10, ptje_test=0.20):
        """
        Funcion que separa data, de manera aleatoria, en set de train y test.
        seed: es la semilla, por default es 10.
        ptje_test: Default= 0.20. Es el valor entre 0 y 1, es la proporcion
        que se quiere dejar para el conjunto test, tomado de self.data.
        """
        random.seed(seed)
        cant_filas_extraer = int(self.data.shape[0] * ptje_test)
        # Crear un vector de números aleatorios entre 0 y len(data)
        cuales = random.sample(range(int(self.data.shape[0])), cant_filas_extraer)
        # datos train:
        self.data_train = self.data.drop(cuales)
        # datos test:
        self.data_test = self.data.iloc[cuales]

        return self.data_test, self.data_train

    def ajustar_modelo(self, x_train, y_train, x_test, y_test):
        """
        Se ajusta el modelo de Regresión Logistica.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # se arma la matriz de diseño agregando la columna de unos
        X = sm.add_constant(self.x_train)
        # se estima el modelo de Regresión Logistica
        modelo = sm.Logit(self.y_train, X)  # esto es lo nuevo
        self.resultado = modelo.fit()
        return self.resultado

    def parametros_modelo(self):
        """
        Retorna las estimaciones de los betas
        del modelo.
        """
        if self.resultado is None:
            print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
            parametros = self.resultado.params
            return parametros

    def ajustados_y(self, prob=0.5):
        """
        Calula el valor predicho a partir del modelo ajustado
        de regresion lineal.
        prob: es el umbral de probabilidad sobre el cual se considera
        para formar el y_ajustado.
        Retorna una tupla: y_ajustado_binary, ajust_y_prob
        """
        if self.resultado is None:
            print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
            X_test = sm.add_constant(self.x_test)
            ajust_y_prob = self.resultado.predict(X_test)
            # probabilidad del "y" ajustado
            # lo que sigue es el "y" ajustado como binario,
            # de acuerdo al "prob" usado:
            y_ajustado_binary = [1 if k >= prob else 0 for k in ajust_y_prob]

            return y_ajustado_binary, ajust_y_prob

    def matriz_confusion(self, prob=0.5):
        """
        Retorna la Matriz de Confusión:
                    |tp     fp|
                    |fn     tn|
        por medio de una lista de la forma [tp, fp, fn, tn]
        prob es la probabilidad.
        """
        if self.resultado is not None:
            y_ajustado = self.ajustados_y(prob)[0]
            n = len(self.y_test)
            tp = 0  # true positive
            tn = 0  # true negative
            fp = 0  # false positive
            fn = 0  # falso negativo
            for i in range(n):
                if y_ajustado[i] == self.y_test.iloc[i]:
                    if y_ajustado[i] == 1:
                        tp = tp + 1
                    else:
                        tn = tn + 1
                else:
                    if y_ajustado[i] == 1 and self.y_test.iloc[i] == 0:
                        fp = fp + 1
                    else:
                        fn = fn + 1
            # print(f" {tp}   {fp} \n {fn}   {tn}")
            return [tp, fp, fn, tn]
        else:
            print("Correr primero ajustados_y()")

    def especif_sensib(self, prob=0.5):
        """
        Retorna una lista con la sensibilidad y especifisidad del modelo
        Regresión Logística, como sigue: [sensibilidad, especificidad].
        prob: es el umbral de probabilidad sobre el cual se determina si una
        respuesta es 0 o 1. Default = 0.5
        """
        matrix_conf = self.matriz_confusion(prob)
        tp = matrix_conf[0]
        fp = matrix_conf[1]
        fn = matrix_conf[2]
        tn = matrix_conf[3]

        sensibilidad = tp / (tp + fn)
        especificidad = tn / (fp + tn)

        # print(f"La sensibilidad del modelo es: {sensibilidad}")
        # print(f"La especificidad del modelo es: {especificidad}")
        return [sensibilidad, especificidad]

    def curva_ROC(self, prob=0.5):
        """
        prob: es el umbral de probabilidad sobre el cual se determina si una
        respuesta es 0 o 1. Default = 0.5
        """
        if self.resultado is not None:
            grid = np.linspace(0, 1, 100)
            l1 = []  # 1-especificidad
            l2 = []  # sensibilidad
            prediccion = self.ajustados_y(prob)[1]
            for j in grid:
                y_pred_binary = [1 if k >= j else 0 for k in prediccion]
                metrica = self.especif_sensib(j)
                especificidad = metrica[1]
                sensibilidad = metrica[0]
                l1.append(1-especificidad)
                l2.append(sensibilidad)

            plt.plot()
            plt.plot(l1, l2, linestyle="-", color="red", label="Curva ROC")
            plt.legend()
            plt.xlabel("1-especificidad")
            plt.ylabel("sensibilidad")
            plt.title('Curva ROC')
            plt.show()

        else:
            print("Falta ajustar el modelo")

    def predict_y(self, x_new, prob=0.5):
        """
        x_new es una lista, con el/los valor/es que se quiere predecir.
        La funcion retorna el valor de predicciòn para un x_new.
        prob: es el umbral de probabilidad sobre el cual se determina si una
         respuesta es 0 o 1. Default = 0.5
        """
        X_new = x_new.copy()
        X_new.insert(0, 1)
        aux = np.dot(self.resultado.params, X_new)
        pred = np.exp(aux)/(1 + np.exp(aux))
        if pred >= prob:
            y_pred_bin = 1
        else:
            y_pred_bin = 0

        return y_pred_bin

    def auc(self, prob=0.5):
        """
        Imprime la evaluacion del clasificador, teniendo en cuenta la tabla
        dada en teoría.
        """
        if self.resultado is not None:
            grid = np.linspace(0, 1, 100)
            especificidad_list = []
            sensibilidad_list = []
            for k in grid:
                metrica = self.especif_sensib(k)
                especificidad = metrica[1]
                sensibilidad = metrica[0]
                especificidad_list.append(1-especificidad)
                sensibilidad_list.append(sensibilidad)

            roc_auc = auc(1-np.array(especificidad_list), sensibilidad_list)
            # print("AUC:", roc_auc)
            if 0.90 < roc_auc <= 1:
                print(f"El clasificador es Excelente, {roc_auc}")
            elif 0.80 < roc_auc <= 0.90:
                print(f"El clasificador es Bueno, {roc_auc}")
            elif 0.70 < roc_auc <= 0.80:
                print(f"El clasificador es Regular, {roc_auc}")
            elif 0.60 < roc_auc <= 0.70:
                print(f"El clasificador es Pobre, {roc_auc}")
            elif 0.50 < roc_auc <= 0.60:
                print(f"El clasificador es Fallido, {roc_auc}")
            else:
                print(f"El clasificador Muy Malo, {roc_auc}")

        else:
            print("Falta ajustar el modelo, usar ajustar_modelo()")

    def modelo_resumen(self):
        """
        Imprime el summary del modelo ajustado
        """
        if self.resultado is not None:
            print(self.resultado.summary())
        else:
            print("Falta ajustar el modelo, usar ajustar_modelo()")

    def p_valor_betas(self, b_i=0, i=1):
        """
        Es una funcion que retorna el p-valor de un test de hipotesis:
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
            t_obs = (coef_xi - b_i) / SE_est[i]

            # el pvalor:
            X = res.model.exog  # para recuperar la matriz de diseño del modelo
            grados_libertad = len(X[:, i]) - 2
            p_valor = 2 * stats.t.sf(abs(t_obs), df=grados_libertad)

            print(f"p-valor: {p_valor}")
            print(f"t_observado: {t_obs}")
