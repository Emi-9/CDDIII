# Ciencia de Datos III:

 Librería creada para el segundo parcial de la materia.

## Descripción:

 Este módulo proporciona clases / funciones / herramientas simples para calculos estadisticos. La estructura y sus respectivas funciones: ⬇️

## Estructura / Características:

- **analisis**:

    - **descriptivo**:
        - `densidad` (estimacion de densidades por kernel)
        - `evalua_histograma` ()
        - `QQplot` (genera un QQ-Plot)

    - **generadora**:
        - `teorica` (genera un conjunto de datos siguiedo una distribucion exacta)
        - `azar` (genera un conjunto muestral de datos siguiendo una distribucion)

- **modelos**:

    - **regresion_lineal_simple**:
        - `calcular_betas` ()
        - `resumen_grafico_simple` ()
        - `predecir_y` ()
        - `estadistico_t_beta1` ()
        - `region_rechazo_beta1` ()
        - `p_valor_beta1` ()
        - `intervalo_confianza_beta1` ()
        - `intervalo_prediccion_y` ()

    - **regresion_lineal_multiple**:
        - `predecir_y` ()
        - `resumen_modelo` ()

    - **regresion_logistica**:
        - `completar` ()

- **utils**:

    - **resumen_numerico**:
        - `ver` (genera un resumen numerico del conjunto de datos)

## Instalación:

 - Cloná el repositorio y usalo.
 - Instalalo con pip
    `pip install git+https://github.com/Emi-9/CDDIII`

## Ejemplo de uso:

- Generar datos al azar que sigan una distribucion normal (o cualquier otro tipo).
- Gráficos cuántil cuántil (QQ-Plot).
- Curva ROC.
- Ajustar y usar modelos de diferentes tipos (principalmente regresion).
- Calcular intervalos de confianza.
- Mas ejemplos e implementaciones en el link de abajo.

## Otros:

- 🚀 [TODO](TODO.md)
- 📚 [Ejemplos de Uso](EjemplosDeUso.py)
