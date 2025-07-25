# Bank Churn Prediction with Machine Learning

Este proyecto utiliza técnicas de machine learning para predecir la fuga de clientes (churn) en un banco, empleando modelos clásicos y avanzados, así como técnicas para el manejo de datos desbalanceados.

## Estructura del Proyecto

- `train_test_models.ipynb`: Notebook principal donde se realiza la limpieza de datos, ingeniería de variables, entrenamiento, optimización de hiperparámetros (Optuna) y evaluación de múltiples modelos, incluyendo:
  - Modelos clásicos: RandomForest, GradientBoosting, ExtraTrees, LogisticRegression, XGBoost, LightGBM
  - Modelos para datos desbalanceados (imblearn): EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier
  - Técnicas de oversampling: SMOTE, ADASYN
  - Curvas ROC y métricas de desempeño

- `tpot_test.ipynb`: Notebook donde se explora la optimización automática de modelos usando TPOT, una herramienta AutoML basada en algoritmos genéticos.

- `requirements.txt`: Lista de dependencias necesarias para reproducir los notebooks.

- `train.csv`, `test.csv`: Datasets originales para entrenamiento y prueba.
- `train_processed.csv`, `train_cleaned.csv`, etc.: Versiones procesadas de los datos.

## Requisitos

- Python 3.8+
- Ver dependencias en `requirements.txt`

## Ejecución

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Abre y ejecuta los notebooks en el orden sugerido:
   - `train_test_models.ipynb`
   - `tpot_test.ipynb`

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

- Autor: Anderson Joseph Ochoa Trujillo, Cristian Andres Villareal Orozco

---
