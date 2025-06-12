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
    - dividir_datos => particiona en train y test.
    - ajustar_modelo =>
    - parametros_modelo =>
    - val_ajustados =>
    - matriz_confusion =>
    - especificidad_sensibilidad =>
    - curva_ROC =>
    - predecir_y =>
    - AUC =>
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
    def dividir_datos(self, seed=10, particion_test=0.20) -> Any:
        """
        Separa los datos al azar (train y test):
        - seed: semilla para replicabilidad.
        - particion_test: es la proporcion que deja para test (valor: 0 - 1).
        """
        random.seed(seed)
        cant_filas_extraer = int(self.data.shape[0] * particion_test)
        pos = random.sample(range(int(self.data.shape[0])),
                            cant_filas_extraer)
        self.data_train = self.data.drop(pos)
        self.data_test = self.data.iloc[pos]

        return self.data_test, self.data_train

    # TODO: anotaciones.
    # TODO: checkear si no hace override de regresion lineal.
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
        Retorna las estimaciones de los ß.
        """
        if self.resultado is not None:
            return self.resultado.params
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def val_ajustados(self, corte=0.5) -> Any:
        """
        Retorna el valor predicho:
        - corte => umbral para decidir si es 0 o 1.
        """
        if self.resultado is not None:
            X_test = sm.add_constant(self.x_test)
            y_ajustado = self.resultado.predict(X_test)
            y_ajustado_binary = [1 if x >= corte else 0 for x in y_ajustado]
            return y_ajustado_binary, y_ajustado
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def matriz_confusion(self, corte=0.5) -> Any:
        """
        Retorna la Matriz de Confusión:
        *            |vp     fp|
        *            |fn     vn|
        - corte => umbral.
        - vp => verdadero positivo.
        - fp => falso positivo.
        - fn => falso negativo.
        - vn => verdadero negativo.
        """
        if self.resultado is not None:
            y_ajustado = self.val_ajustados(corte)[0]
            n = len(self.y_test)
            vp = 0
            fp = 0
            fn = 0
            vn = 0
            for i in range(n):
                if y_ajustado[i] == self.y_test.iloc[i]:
                    if y_ajustado[i] == 1:
                        vp = vp + 1
                    else:
                        vn = vn + 1
                else:
                    if y_ajustado[i] == 1 and self.y_test.iloc[i] == 0:
                        fp = fp + 1
                    else:
                        fn = fn + 1
            return [vp, fp, fn, vn]
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def especificidad_sensibilidad(self, corte=0.5) -> Any:
        """
        Retorna la sensibilidad y especifisidad del modelo
        - corte => umbral de probabilidad sobre el cual se determina si una
        respuesta es 0 o 1.
        """
        matrix_conf = self.matriz_confusion(corte)
        vp = matrix_conf[0]
        fp = matrix_conf[1]
        fn = matrix_conf[2]
        vn = matrix_conf[3]

        sensibilidad = vp / (vp + fn)
        especificidad = vn / (fp + vn)

        return (sensibilidad, especificidad)

    # TODO: anotaciones.
    def curva_ROC(self, corte=0.5) -> Any:
        """
        corte => umbral.
        """
        if self.resultado is not None:
            grid = np.linspace(0, 1, 100)
            l1 = []
            l2 = []
            prediccion = self.val_ajustados(corte)[1]
            for j in grid:
                y_pred_binary = [1 if k >= j else 0 for k in prediccion]
                metrica = self.especificidad_sensibilidad(j)
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
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def predecir_y(self, x_new, corte=0.5) -> Any:
        """
        Retorna ŷ a partir de un x_new:
        - x_new => valores que se quieren predecir.
        - corte => umbral.
        """
        X_new = x_new.copy()
        X_new.insert(0, 1)
        aux = np.dot(self.resultado.params, X_new)
        pred = np.exp(aux)/(1 + np.exp(aux))

        if pred >= corte:
            y_pred_bin = 1
        else:
            y_pred_bin = 0

        return y_pred_bin

    # TODO: anotaciones.
    # TODO: implementarlo de otra manera, los prints son horribles
    def AUC(self) -> Any:
        """
        Imprime la evaluacion del clasificador.
        """
        if self.resultado is not None:
            grid = np.linspace(0, 1, 100)
            especificidad_list = []
            sensibilidad_list = []
            for k in grid:
                metrica = self.especificidad_sensibilidad(k)
                especificidad = metrica[1]
                sensibilidad = metrica[0]
                especificidad_list.append(1-especificidad)
                sensibilidad_list.append(sensibilidad)

            roc_auc = auc(1-np.array(especificidad_list), sensibilidad_list)
            if 0.90 < roc_auc <= 1:
                print(f"El clasificador es excelente, {roc_auc}")
            elif 0.80 < roc_auc <= 0.90:
                print(f"El clasificador es bueno, {roc_auc}")
            elif 0.70 < roc_auc <= 0.80:
                print(f"El clasificador es regular, {roc_auc}")
            elif 0.60 < roc_auc <= 0.70:
                print(f"El clasificador es pobre, {roc_auc}")
            elif 0.50 < roc_auc <= 0.60:
                print(f"El clasificador es fallido, {roc_auc}")
            else:
                print(f"El clasificador es muy malo, {roc_auc}")
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def modelo_resumen(self) -> Any:
        """
        Imprime el summary del modelo ajustado.
        """
        if self.resultado is not None:
            print(self.resultado.summary())
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")

    # TODO: anotaciones.
    def p_valor_betas(self, b_i: float = 0, i: int = 1) -> Any:
        """
        Retorna el p-valor del test de hipotesis:
        * H_0: beta_i = k vs H_1 beta_i != k
        - b_i => numero k sobre el cual se quiere hacer el test.
        - i => indice del beta que se quiere testear (es un entero).
        """
        if self.resultado is not None:
            res = self.resultado
            SE_est = res.bse
            coef_xi = res.params[i]
            t_obs = (coef_xi - b_i) / SE_est[i]

            X = res.model.exog
            grados_libertad = len(X[:, i]) - 2
            p_valor = 2 * stats.t.sf(abs(t_obs), df=grados_libertad)

            print(f"p-valor: {p_valor}")
            print(f"t_observado: {t_obs}")
        else:
            raise RuntimeError("AJUSTAR EL MODELO ANTES")
