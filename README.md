# Predicción de Esperanza de Vida — Banco Mundial / OMS

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0.0-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-FF6600)
![SHAP](https://img.shields.io/badge/SHAP-latest-8A2BE2)
![numpy](https://img.shields.io/badge/numpy-2.4.1-013243?logo=numpy&logoColor=white)

Modelo predictivo de esperanza de vida al nacer por país y año, desarrollado como ejercicio de consultoría con datos del **Banco Mundial** y la **OMS** · 2026 · Autores: David Rodríguez y Juan Rueda.

---

## Problema

El Banco Mundial necesita predecir la esperanza de vida al nacer para 193 países a partir de indicadores socioeconómicos y de salud pública, con dos objetivos simultáneos: maximizar la precisión predictiva y mantener la interpretabilidad suficiente para orientar decisiones de política de desarrollo.

---

## Datos

**Fuente:** [Life Expectancy (WHO) — Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

- **Cobertura:** 182 países · 2000–2015 · 2 912 observaciones país-año
- **Variables originales:** 22 (mortalidad adulta e infantil, prevalencia de VIH/SIDA, coberturas de vacunación, gasto en salud, PIB per cápita, escolaridad, IDH)
- **Variable objetivo:** Esperanza de vida al nacer (rango 36–89 años)

**Limpieza aplicada:** imputación manual con series del Banco Mundial para 31 países, interpolación temporal, k-NN por clústeres (k=10) para 1 426 valores faltantes y detección/corrección de 103 valores atípicos.

---

## Metodología

**Pipeline de preprocesamiento (5 etapas):** limpieza de panel, imputación manual, interpolación temporal, k-NN por clúster (K-Means k=10) y corrección de outliers (criterio IQR dual).

**Feature engineering (5 variables nuevas):**

| Variable | Transformación | Ganancia en correlación |
|----------|---------------|------------------------|
| `log_hiv_aids` | log(HIV/SIDA) | r: −0.558 → −0.764 |
| `log_gdp_per_capita` | log(PIB per cápita) | r: 0.431 → 0.562 |
| `thinness_avg` | Media de delgadez 5–19 años | Reduce multicolinealidad (r=0.94) |
| `vaccination_index` | Media de 3 vacunas | r: 0.175 → 0.459 |
| `postneonatal_deaths` | Muertes bajo-5 − infantiles | Captura mortalidad 1–4 años |

**Validación temporal:** train 2000–2012 (2 366 obs.) / test 2013–2015 (546 obs.), preservando la estructura de panel.

**Modelos:** Elastic Net (ElasticNetCV), Random Forest (RandomForestRegressor), XGBoost (XGBRegressor). Ajuste de hiperparámetros con `RandomizedSearchCV` (5-fold CV, métrica RMSE). Interpretabilidad con SHAP. Reducción de dimensionalidad exploratoria con PCA.

---

## Hallazgos principales

| Modelo | RMSE (test) | R² (test) |
|--------|-------------|-----------|
| Elastic Net | 2.873 años | 0.880 |
| Random Forest | 2.115 años | 0.935 |
| **XGBoost** | **1.998 años** | **0.942** |

El modelo **XGBoost** explica el **94.2 % de la varianza** entre países con un error promedio inferior a 2 años, suficiente para estratificación de brechas de desarrollo y escenarios contrafactuales de política.

Según el análisis SHAP, la **composición de ingresos del IDH** tiene 2.2× mayor impacto que la mortalidad adulta y 2.5× mayor que la prevalencia de VIH/SIDA, lo que sugiere que las intervenciones estructurales sobre el ingreso generan mayores retornos en longevidad que las intervenciones sanitarias puntuales.

---

## Estructura del repositorio

```
Prediccion_Esperanza_Vida_Banco_Mundial/
├── Punto2_Regresion.ipynb       # Análisis completo y modelos
├── Informe_BancoMundial.pdf     # Informe ejecutivo (2 páginas)
├── Enunciado.pdf                # Especificación del proyecto
├── requirements.txt
├── Datos/
│   ├── Life Expectancy Data.csv # Dataset original (Kaggle/OMS)
│   └── life_expectancy_clean.csv
├── Modelos/
│   └── Punto2_modelo.pkl        # Modelo XGBoost serializado
└── Visualizaciones/
    ├── pca_biplot.png
    ├── shap_importance_xgb.png
    └── shap_beeswarm_xgb.png
```
