# from statsmodels.nonparametric.kde import KDEUnivariate
import numpy as np
from typing import Any
from scipy.stats import norm, uniform


class GeneradoraDeDatos:
    '''
    Primero se instancia (toma la cantidad a generar), luego:
    - teorica => exacta
    - azar => muestral
    '''

    def __init__(self, q: int) -> None:
        self.q = q

    def __main__(self):
        pass

    def teorica(self, x: list[Any] | np.ndarray[Any, Any],
                mu: float = 0, sigma: float = 1,
                type: str = "normal") -> np.ndarray[Any, Any]:
        '''
        - x => cualquier iterable que contenga los puntos donde
               se va a evaluar.
        - mu => punto sobre el cual va a estar centrada la
                distribucion (media).
        - sigma => amplitud de la distribucion (varianza).
        - type => normal, uniforme o BS.
        '''
        if type == "normal":
            return norm.pdf(x, mu, sigma)
        elif type == "uniforme":
            return uniform.pdf(x, mu, sigma)
        elif type == "BS":
            term1 = (1/2) * norm.pdf(x, loc=0, scale=1)
            term_mid = [norm.pdf(x, loc=j/2-1, scale=1/10) for j in range(5)]
            term2 = (1/10) * sum(term_mid)
            return term1 + term2
        else:
            print("Tipo INCORRECTO")
            return np.array([])  # Callate pylance

    def azar(self, mu: float = 0, sigma: float = 1,
             type: str = "normal") -> np.ndarray[Any, Any] | int:
        # Segun pylance existe la posibilidad de que retorne int
        # Al menos asi se calla
        '''
        - x => cualquier iterable que contenga los puntos donde
               se va a evaluar.
        - mu => punto sobre el cual va a estar centrada la
                distribucion (media).
        - sigma => amplitud de la distribucion (varianza).
        - type => normal, uniforme o BS.
        '''
        if type == "normal":
            return norm.rvs(loc=mu, scale=sigma, size=self.q)
        elif type == "uniforme":
            return uniform.rvs(loc=mu, scale=sigma, size=self.q)
        elif type == "BS":
            u = np.random.uniform(size=(self.q))
            y = u.copy()
            ind = np.where(u > 0.5)[0]
            y[ind] = np.random.normal(0, 1, size=len(ind))
            for j in range(5):
                ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
                y[ind] = np.random.normal(j/2-1, 1/10, size=len(ind))
            return y
        else:
            print("Tipo INCORRECTO")
            return np.array([])  # Callate pylance
