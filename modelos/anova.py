import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from scipy import stats
from analisis.descriptivo import AnalisisDescriptivo


def anova(df, response_col, factor_col, alpha=0.05):
    """
    Realiza un análisis ANOVA completo sobre un DataFrame.

    Parámetros:
    - df            : pandas.DataFrame con los datos.
    - response_col  : nombre de la columna respuesta (string).
    - factor_col    : nombre de la columna factor (string).
    - alpha         : nivel de significación para tests e intervalos (default 0.05).

    Extra:
    - Armar el DataFrame asi
    * respuesta = [datos]
    * factores = [factor1] * (cantidad_factor1) + [factorn] * (cantidad_factorn)
    """

    # 1. Boxplot para detectar diferencias
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=factor_col, y=response_col, data=df)
    plt.title(f'Boxplot de {response_col} por {factor_col}')
    plt.xlabel(factor_col)
    plt.ylabel(response_col)
    plt.show()
    print("Boxplot creado: inspecciona visualmente diferencias entre niveles.")

    # 2. Ajuste del modelo completo
    formula = f"{response_col} ~ C({factor_col})"
    modelo_full = smf.ols(formula, data=df).fit()
    print("=== Modelo completo ajustado ===")
    print(modelo_full.summary())

    # Ecuación ajustada
    params = modelo_full.params
    eq = f"Modelo: \u03C8 = {params['Intercept']:.4f}"
    for name, coef in params.items():
        if name != 'Intercept':
            eq += f" + ({coef:.4f})*{name}"
    print(eq)
    print("(Comparar coeficientes y p-valores de dummies con 0 para ver efecto.)\n")

    # 3. Diagnóstico de residuos
    residuos = modelo_full.resid
    predichos = modelo_full.fittedvalues

    # 3a) Residuos vs predichos
    plt.figure(figsize=(6, 4))
    plt.scatter(predichos, residuos)
    plt.axhline(0, linestyle='--')
    plt.title('Residuos vs. Valores Predichos')
    plt.xlabel('Predichos')
    plt.ylabel('Residuos')
    plt.show()
    print("Gráfico residuos vs predichos: detecta heterocedasticidad.")

    # 3b) Breusch–Pagan
    bp = sm.stats.diagnostic.het_breuschpagan(residuos, modelo_full.model.exog)
    p_bp = bp[1]
    print(f"Breusch–Pagan p-valor: {p_bp:.4f}")
    if p_bp < alpha:
        print(f"→ p-valor < {alpha}: RECHAZAR H0, hay heterocedasticidad.")
    else:
        print(f"→ p-valor ≥ {alpha}: NO rechazar H0, homocedasticidad aceptable.")

    # 3c) QQ-plot y Shapiro
    sm.qqplot(residuos, line='45')
    plt.title('QQ-plot de residuos')
    plt.show()
    AnalisisDescriptivo(residuos).QQplot()
    stat_sw, p_sw = stats.shapiro(residuos)
    print(f"Shapiro–Wilk p-valor: {p_sw:.4f}")
    if p_sw < alpha:
        print(f"→ p-valor < {alpha}: RECHAZAR H0, residuos no normales.")
    else:
        print(f"→ p-valor ≥ {alpha}: NO rechazar H0, normalidad aceptable.")

    # 4. ANOVA vs modelo nulo
    modelo_null = smf.ols(f"{response_col} ~ 1", data=df).fit()
    print("\n=== ANOVA: nulo vs completo ===")
    anova_res = anova_lm(modelo_null, modelo_full)
    print(anova_res)
    # Obtener p-valor de la comparación en la segunda fila
    p_glob = anova_res['Pr(>F)'][1]
    print(f"p-valor global ANOVA: {p_glob:.4f}")
    if p_glob < alpha:
        print(f"→ p-valor < {alpha}: RECHAZAR H0, el factor es significativo.")
    else:
        print(f"→ p-valor ≥ {alpha}: NO rechazar H0, no hay efecto global.")

    # 4b) Cálculo manual F
    RSS_m = sum(modelo_null.resid**2)
    RSS_M = sum(modelo_full.resid**2)
    n = df.shape[0]
    p_full = len(modelo_full.params)
    p_null = len(modelo_null.params)
    df1 = p_full - p_null
    df2 = n - p_full
    F_obs = ((RSS_m - RSS_M)/df1) / (RSS_M/df2)
    p_man = 1 - stats.f.cdf(F_obs, df1, df2)
    F_crit = stats.f.ppf(1 - alpha, df1, df2)
    print(f"\nF observado: {F_obs:.4f}")
    print(f"F crítico (α={alpha}): {F_crit:.4f}")
    if F_obs > F_crit:
        print(f"→ F_obs > F_crit: RECHAZAR H0, factor significativo.")
    else:
        print(f"→ F_obs ≤ F_crit: NO rechazar H0, no hay efecto.")
    print(f"p-valor manual: {p_man:.4f}")

    # 5. Intervalo de confianza entre dos niveles
    niveles = df[factor_col].unique()
    if len(niveles) >= 2:
        lvl1, lvl2 = niveles[:2]
        # Cálculo de medias
        m1 = df[df[factor_col] == lvl1][response_col].mean()
        m2 = df[df[factor_col] == lvl2][response_col].mean()
        # Error estándar usando MSE del modelo completo
        mse = RSS_M / df2
        n1 = df[df[factor_col] == lvl1].shape[0]
        n2 = df[df[factor_col] == lvl2].shape[0]
        se = (mse * (1/n1 + 1/n2))**0.5
        # t crítico
        t_crit = stats.t.ppf(1 - alpha/2, df2)
        me = t_crit * se
        # Intervalo de confianza
        ic = (m1 - m2 - me, m1 - m2 + me)
        conf = 100 * (1 - alpha)
        print(f"\nIC {conf:.1f}% para ({lvl1} - {lvl2}): {ic}")

        # Validación del intervalo
        lim_inf, lim_sup = ic
        if lim_inf > 0 or lim_sup < 0:
            print(f"Como el intervalo de confianza al {conf:.1f}% ({lim_inf:.3f}, {lim_sup:.3f}) no contiene el 0, "
                  f"es posible concluir con un {conf:.1f}% de confianza que la media de '{lvl1}' difiere de la media de '{lvl2}'.")
        else:
            print(f"Como el intervalo de confianza al {conf:.1f}% ({lim_inf:.3f}, {lim_sup:.3f}) contiene el 0, "
                  f"no se puede concluir con un {conf:.1f}% de confianza que las medias de '{lvl1}' y '{lvl2}' difieran significativamente.")
    return residuos, modelo_full
