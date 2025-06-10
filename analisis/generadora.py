# from statsmodels.nonparametric.kde import KDEUnivariate
import numpy as np
from scipy.stats import norm, uniform


class GeneradoraDeDatos:
    '''
    Documentar
    '''

    def __init__(self, q: int) -> None:
        self.q = q

    def __main__(self):
        pass

    def teorica(self, x, mu, sigma, type):
        '''
        Documentar
        '''
        if type == "normal":
            return norm.pdf(x, mu, sigma)
        elif type == "uniforme":
            return uniform.pdf(x, mu, sigma)
        elif type == "BS":
            print("es una distribucion inventada por las profesora"
                  "no se como es la teorica")
        else:
            print("Tipo INCORRECTO")

    def azar(self, mu, sigma, type):  # TODO: completar
        '''
        Documentar
        '''
        res = uniform.rvs(loc=mu, scale=sigma, size=self.q)
        return res

    def norm(self, mu, sigma):  # combinar con la de azar
        res = norm.rvs(loc=mu, scale=sigma, size=self.q)
        return res

    def BS(self):
        '''
        Bart Simpson.
        '''
        u = np.random.uniform(size=(self.q))
        y = u.copy()
        ind = np.where(u > 0.5)[0]
        y[ind] = np.random.normal(0, 1, size=len(ind))
        for j in range(5):
            ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
            y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))
        return y
