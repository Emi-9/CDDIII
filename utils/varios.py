import numpy as np
from typing import Any


class resumen_numerico:
    '''
    Documentar
    '''

    def __init__(self, data: list[Any] | np.ndarray[Any, Any]) -> None:
        self.data = data

    def __main__(self):
        pass

    def ver(self) -> dict[str, Any]:
        return {
            "Media": np.mean(self.data),
            "Mediana": np.median(self.data),
            "STD": np.std(self.data),
            "Minimo": np.min(self.data),
            "Maximo": np.max(self.data),
            "Cuartiles": np.percentile(self.data, q=[25, 50, 75]),
            "Cantidad": len(self.data)
        }
