import numpy as np
from typing import Any


class AnalisisDescriptivo:
    '''Documentar'''

    def __init__(self, data: list[Any] | np.ndarray[Any, Any]) -> None:
        self.data = data

    def __main__(self):
        pass

    def _kernel_gaussiano(self, u: Any) -> float:  # TODO
        # Ver verdaderamente lo que se espera que sea "u"
        # unicamente np.ndarray (??) (checkear) (list => bug?)
        valor_kernel_gaussiano = (1/np.sqrt(2*np.pi)) * np.e ** ((-1/2)*(u**2))
        return sum(valor_kernel_gaussiano)

    def _kernel_uniforme(self, u: Any) -> float:  # TODO
        valor_kernel_uniforme = (u >= -1/2) & (u <= 1/2)
        return sum(valor_kernel_uniforme)

    def _kernel_cuadratico(self, u: Any) -> float:  # TODO
        a = 3/4 * (1-u**2)
        b = (u >= -1) & (u <= 1)
        return sum(a*b)

    def _kernel_triangular(self, u: Any) -> float:  # TODO
        a = (1+u) * ((u >= -1) & (u <= 0))
        b = (1-u) * ((u >= 0) & (u <= 1))
        return sum(a+b)

    def densidad(self, x: list[Any] | np.ndarray[Any, Any], h: int,
                 kernel: str = "uniforme") -> np.ndarray[Any, Any]:
        '''Documentar'''

        n = len(self.data)
        density = np.zeros(len(x))

        for i in range(len(x)):
            u = (self.data - x[i]) / h
            if kernel == "gaussiano":
                res = self._kernel_gaussiano(u)
            elif kernel == "uniforme":
                res = self._kernel_uniforme(u)
            elif kernel == "cuadratico":
                res = self._kernel_cuadratico(u)
            elif kernel == "triangular":
                res = self._kernel_triangular(u)
            else:
                print("El tipo de kernel especificado es INCORRECTO")
                break
            density[i] += res / (n*h)

        return density

    def evalua_histograma(self, h: int,
                          x: list[Any] | np.ndarray[Any, Any]
                          ) -> np.ndarray[Any, Any]:
        '''Documentar'''

        bins = np.arange(min(self.data)-h, max(self.data)+h+1, h)
        frec = np.zeros(len(bins)-1)
        res = np.zeros(len(x))

        for y in range(len(self.data)):
            for i in range(len(bins)-1):
                if self.data[y] >= bins[i] and self.data[y] < bins[i+1]:
                    frec[i] += (1 / len(self.data)) / h
                    break

        for y in range(len(x)):
            for i in range(len(bins)-1):
                if x[y] >= bins[i] and x[y] < bins[i+1]:
                    res[y] = frec[i]
                    break

        return res
