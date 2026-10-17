
# Bank Churn Prediction with Machine Learning

Este proyecto utiliza técnicas de machine learning para predecir la fuga de clientes (churn) en un banco, empleando modelos clásicos y avanzados, así como técnicas para el manejo de datos desbalanceados.

El trabajo se basa en la competencia de Kaggle: [Bank Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge).

## Estructura del Proyecto

- `Notebooks/`:
  - `train_test_models.ipynb`: Notebook principal con protocolo 100% libre de fugas de datos (Data Leakage), validación cruzada estratificada de 5 particiones (5-Fold Stratified CV), evaluación comparativa de 13 modelos y ensamble limpio, curvas ROC, matrices de confusión y explicabilidad SHAP.
  - `EDA.ipynb`: Análisis exploratorio de datos, distribuciones y correlaciones.
  - `tpot_test.ipynb`: Búsqueda automática con TPOT basada en algoritmos genéticos.

- `Datasets/`:
  - `train.csv`, `test.csv`: Datasets originales.
  - `train_processed.csv`: Dataset procesado sin identificadores, listo para modelado.

- `Figs/`:
  - `ROC_Curvas.png`: Curvas ROC comparativas sobre el conjunto de prueba aislado.
  - `shap_summary.png`: Gráfico Beeswarm de explicabilidad global con SHAP.
  - `cm/confusion_matrices_comparison.png`: Matrices de confusión normalizadas de los modelos clave.

- `Submissions/`:
  - Contiene las 14 submissions independientes numeradas del `01` al `14`, generadas mediante 5-Fold Bagging sobre el 100% de los datos de entrenamiento para evaluar en Kaggle.

- `Review/`:
  - Manuscrito y documentación de referencia.

- `archive/`:
  - Respaldo de artefactos, figuras y submissions anteriores fuera del flujo de trabajo activo.

- `requirements.txt`: Lista de dependencias del entorno.

## Descripción General

El flujo principal consiste en:
- Preprocesamiento y codificación de variables categóricas
- División de datos en entrenamiento y prueba
- Optimización de hiperparámetros con Optuna
- Entrenamiento y evaluación de modelos clásicos y de imblearn
- Comparación de desempeño mediante curvas ROC y métricas como AUC, accuracy, sensibilidad, especificidad y G-mean
- Uso de TPOT para búsqueda automática de pipelines óptimos

## Resultados

Los resultados incluyen:
- Métricas de desempeño para cada modelo
- Curvas ROC comparativas
- Archivos de predicción para submit

## Créditos

- Autores: Anderson Joseph Ochoa Trujillo, Cristian Andres Villareal Orozco, Juan Pablo Hoyos Sanchez

---


