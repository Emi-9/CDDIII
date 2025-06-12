import numpy as np
import random
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from typing import Any

# TODO: terminar de adaptar a mi estilo
# ^ (usabilidad y nombres de funciones mas sencillos)


class RegresionLogistica:
    """
    Primero se instancia (), luego:
    - separar_data_train_test =>
    - ajustar_modelo =>
    - parametros_modelo =>
    - ajustados_y =>
    - matriz_confusion =>
    - especif_sensib =>
    - curva_ROC =>
    - predict_y =>
    - auc =>
    - modelo_resumen =>
    - p_valor_betas =>
    """

    def __init__(self, data: Any) -> None:  # TODO: anotaciones
        self.data = data
        self.data_train = None
        self.data_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.resultado = None

    def __main__(self):
        pass

    # TODO: anotaciones
    def separar_data_train_test(self, seed=10, ptje_test=0.20) -> Any:
        """
        Separa los datos al azar (train y test):
        - seed: semilla para replicabilidad.
        - ptje_test: valor entre 0 y 1, es la proporcion
                     que deja para el conjunto test (tomado de self.data).
        """
        random.seed(seed)
        cant_filas_extraer = int(self.data.shape[0] * ptje_test)
        cuales = random.sample(range(int(self.data.shape[0])),
                               cant_filas_extraer)
        self.data_train = self.data.drop(cuales)
        self.data_test = self.data.iloc[cuales]

        return self.data_test, self.data_train

    # TODO: anotaciones
    def ajustar_modelo(self, x_train, y_train, x_test, y_test) -> Any:
        """
        Ajusta el modelo.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        X = sm.add_constant(self.x_train)
        modelo = sm.Logit(self.y_train, X)
        self.resultado = modelo.fit()

        return self.resultado

    # TODO: anotaciones
    def parametros_modelo(self) -> Any:
        """
        Retorna las estimaciones de los betas.
        """
        if self.resultado is not None:
            return self.resultado.params
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def ajustados_y(self, prob=0.5) -> Any:
        """
        Calula el valor predicho a partir del modelo ajustado
        de regresion lineal.
        prob: es el umbral de probabilidad sobre el cual se considera
        para formar el y_ajustado.
        Retorna una tupla: y_ajustado_binary, ajust_y_prob
        """
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
        else:
            X_test = sm.add_constant(self.x_test)
            ajust_y_prob = self.resultado.predict(X_test)
            y_ajustado_binary = [1 if k >= prob else 0 for k in ajust_y_prob]

            return y_ajustado_binary, ajust_y_prob

    # TODO: anotaciones.
    def matriz_confusion(self, prob=0.5) -> Any:
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
            tp = 0
            tn = 0
            fp = 0
            fn = 0
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
            return [tp, fp, fn, tn]
        else:
            print("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def especif_sensib(self, prob=0.5) -> Any:
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

        return (sensibilidad, especificidad)

    # TODO: anotaciones.
    def curva_ROC(self, prob=0.5) -> Any:
        """
        prob: es el umbral de probabilidad sobre el cual se determina si una
        respuesta es 0 o 1.
        """
        if self.resultado is not None:
            grid = np.linspace(0, 1, 100)
            l1 = []
            l2 = []
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
            plt.title("Curva ROC")
            plt.show()
        else:
            print("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def predict_y(self, x_new, prob=0.5) -> Any:
        """
        x_new es una lista, con el/los valor/es que se quiere predecir.
        La funcion retorna el valor de predicciòn para un x_new.
        prob: es el umbral de probabilidad sobre el cual se determina si una
         respuesta es 0 o 1.
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

    # TODO: anotaciones.
    def auc(self, prob=0.5) -> Any:
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
            print("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def modelo_resumen(self) -> Any:
        """
        Imprime el summary del modelo ajustado
        """
        if self.resultado is not None:
            print(self.resultado.summary())
        else:
            print("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def p_valor_betas(self, b_i=0, i=1) -> Any:
        """
        Es una funcion que retorna el p-valor de un test de hipotesis:
        H_0: beta_i = k vs H_1 beta_i != k
        b_i: es el numero k sobre el cual se quiere hacer el test.
        i: es el indice del beta que se quiere testear (es un entero).
        """
        if self.resultado is None:
            print("AJUSTAR EL MODELO ANTES")
        else:
            res = self.resultado
            SE_est = res.bse
            coef_xi = res.params[i]
            t_obs = (coef_xi - b_i) / SE_est[i]

            X = res.model.exog
            grados_libertad = len(X[:, i]) - 2
            p_valor = 2 * stats.t.sf(abs(t_obs), df=grados_libertad)

            print(f"p-valor: {p_valor}")
            print(f"t_observado: {t_obs}")
