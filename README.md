# Financial Bridge MX: Robust & Explainable Credit Risk System

[![XAI: Shapley](https://img.shields.io/badge/XAI-Shapley%20Values-green.svg)](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76cda3a0a00f8c22-Abstract.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema de evaluación de riesgo crediticio diseñado para eliminar la opacidad de los modelos de "caja negra". Implementa **Shapley Values** para atribución de características, protocolos de **Robustez** ante ruido y mitigación de **Sesgo (Fairness)**.

##  Dataset
El proyecto utiliza el **LendingClub Dataset**, enfocado en créditos aprobados y defaulteados.
* **Source:** [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
* **Pre-processing:** Se filtran 50,000 registros con variables clave como `dti`, `annual_inc` y `revol_util`.



##  Módulos del Sistema
* **M1 (XAI):** Atribución local mediante Monte Carlo Shapley Values.
* **M2 (Robustness):** Estabilidad validada bajo ruido Gaussiano ($\sigma=0.02$).
* **M3 (Fairness):** Reducción de impacto dispar mediante reweighting de grupos sensibles.
* **M4 (Governance):** Logging determinista en `run_log.json` y reproducibilidad (Seed 42).

##  Tech Stack
* **Model:** Random Forest Classifier (Scikit-Learn).
* **Interpretability:** Custom Monte Carlo Shapley Implementation.
* **Metrics:** AUC (0.7503), Accuracy (0.7306), Demographic Parity, Equal Opportunity.

##  Ejecución
1. **Abrir Notebook:** [Financial Bridge MX en Colab](https://colab.research.google.com/drive/10Xzv7-WmO22f4png0_brfOs5PWqJ34Mb)
2. **Ejecutar:** El script procesa el dataset y genera la carpeta `midterm2_outputs/` con los siguientes artefactos:
   - `shap_cliente0.png` (Explicación local)
   - `stability_std.png` (Prueba de robustez)
   - `fairness_compare.png` (Benchmark ético)

---
**Integrantes:** Iván Ornelas, Luis Serrano, Guillermo Lira, Victor Bosquez.  
**Materia:** Análisis y diseño de algoritmos avanzados - Tecnológico de Monterrey.
