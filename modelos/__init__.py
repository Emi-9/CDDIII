from .regresion_lineal import RegresionLineal
from .regresion_lineal_simple import RegresionLinealSimple
from .regresion_lineal_multiple import RegresionLinealMultiple
from .regresion_logistica import RegresionLogistica
from .anova import generic_anova_analysis as anova

__all__ = ["RegresionLineal", "RegresionLinealSimple",
           "RegresionLinealMultiple", "RegresionLogistica", "anova"]
