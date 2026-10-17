# Bank Customer Churn Prediction with Machine Learning and AutoML

This repository provides an end-to-end, leak-free machine learning benchmark for predicting bank customer churn, evaluating competitive ensemble architectures, class imbalance strategies, and automated pipeline optimization (AutoML).

The project is developed using the Kaggle benchmark: [Bank Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge).

---

## Key Methodological Features

- **Strict Zero Data Leakage Protocol:** Partitioned via an 80/20 stratified split (`X_train, y_train` vs. sequestered `X_test, y_test`). All preprocessing transformers (`StandardScaler`, `OneHotEncoder`) and synthetic resamplers (`SMOTE`, `ADASYN`) are encapsulated within `imblearn.pipeline.Pipeline` objects fitted strictly on training folds.
- **Stratified 5-Fold Cross-Validation:** Hyperparameter optimization and validation are strictly performed on training folds via Optuna (TPE Sampler) and TPOT evolutionary search.
- **Imbalance Mitigation Matrix:** Evaluates baseline linear models, algorithmic cost-sensitive learners (`BalancedRandomForest`, `EasyEnsemble`, `RUSBoost`), synthetic oversamplers (`SMOTE`, `ADASYN`), gradient boosting architectures (`XGBoost`, `LightGBM`, `CatBoost`, `HistGradientBoosting`), and ensemble blends.
- **Genetic AutoML Synthesis:** TPOT genetic programming pipeline search ($G=15, N=40$) discovering non-linear feature selectors and stacking ensembles.
- **Model Explainability:** Tree SHAP feature attribution (global summary beeswarm and local waterfall plots) uncovering non-linear churn drivers without baseline truncation.
- **Publication Figures:** High-resolution (300 DPI) publication-ready plots formatted for academic journals (clean axes, diagonal inset zooms, multi-panel confusion matrix composite).

---

## Repository Structure

```text
├── Datasets/
│   ├── train.csv                      # Raw Kaggle training dataset (15,000 samples)
│   ├── test.csv                       # Raw Kaggle test dataset (10,000 samples)
│   └── train_processed.csv            # Preprocessed dataset ready for modeling
├── Notebooks/
│   ├── train_test_models.ipynb        # Primary leak-free training, CV, Optuna HPO, evaluation & SHAP
│   ├── tpot_test.ipynb                # TPOT genetic AutoML evolutionary pipeline search
│   └── EDA.ipynb                      # Exploratory data analysis, distributions, and correlations
├── Figs/
│   ├── ROC_Mejores_Modelos.png        # Comparative ROC curves for Top-performing models (with zoom inset)
│   ├── ROC_Menor_Rendimiento.png      # Comparative ROC curves for lower-performing models (with zoom inset)
│   ├── cm/
│   │   ├── figure_6_confusion_matrices_panel.png  # 5-panel composite confusion matrix (Top 5 models)
│   │   ├── 01_cm_xgboost.png
│   │   ├── 02_cm_hist_gradient_boosting.png
│   │   ├── 03_cm_clean_ensemble_top3.png
│   │   ├── 04_cm_lightgbm_adasyn.png
│   │   └── 05_cm_tpot_pipeline.png
│   ├── shap_summary_xgboost.png       # Global SHAP beeswarm feature importance plot
│   └── shap_waterfall_example.png     # Local patient-level SHAP waterfall explanation
├── Submissions/                       # Kaggle prediction submissions (01-14 + TPOT)
├── best_models_hyperparameters.json   # Machine-readable optimal hyperparameters and performance metrics
├── tpot_best_pipeline.py              # Exported production pipeline for TPOT's best evolutionary model
├── archive/                           # Legacy submissions, figures, and exploratory backups
├── requirements.txt                   # Environment dependencies
└── README.md
```

---

## Top Models Performance Summary

All models were evaluated using Stratified 5-Fold Cross-Validation on the 80% training set (12,000 instances) and tested on the isolated 20% holdout test set (3,000 instances):

| Rank | Model | Imbalance Strategy | CV ROC-AUC | Test ROC-AUC | Accuracy | Recall | Specificity | F1-Score | Kaggle AUC |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **XGBoost** | Cost-Insensitive Booster | $0.9348 \pm 0.0053$ | **0.9340** | 0.8947 | 0.6650 | 0.9535 | 0.7204 | **0.94158** |
| **2** | **HistGradientBoosting** | Histogram Booster | $0.9327 \pm 0.0045$ | 0.9326 | 0.8987 | 0.6781 | 0.9552 | 0.7319 | **0.94132** |
| **3** | **Ensemble Blend (Top 3)** | Multi-Model Probability Average | $0.9363 \pm 0.0045$ | **0.9343** | 0.8983 | 0.6699 | 0.9569 | 0.7289 | **0.94129** |
| **4** | **LightGBM + ADASYN** | Adaptive Synthetic Resampling | $0.9327 \pm 0.0038$ | 0.9323 | 0.8843 | **0.7533** | 0.9179 | 0.7266 | **0.94073** |
| **5** | **TPOT AutoML Pipeline** | Genetic Search (Stacking + Perc64 + GBDT) | $0.9363 \pm 0.0039$ | 0.9323 | **0.8953** | 0.6732 | 0.9523 | **0.7241** | **0.94060** |

---

## Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Joseph0choa/Bank_Churn_ML.git
   cd Bank_Churn_ML
   ```

2. **Create a virtual environment (conda recommended):**
   ```bash
   conda create -n churn_env python=3.11 -y
   conda activate churn_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Reproduction Guide

1. **Model Training & Evaluation:**
   Open and execute [`Notebooks/train_test_models.ipynb`](Notebooks/train_test_models.ipynb). By default, pre-tuned optimal hyperparameters are loaded for immediate execution. To rerun Optuna Bayesian optimization studies, set `RUN_OPTUNA_SEARCH = True`.

2. **Automated Machine Learning (TPOT):**
   Open [`Notebooks/tpot_test.ipynb`](Notebooks/tpot_test.ipynb) to inspect or execute the evolutionary pipeline optimization over 15 generations.

3. **Exported Pipeline:**
   To run predictions directly with TPOT's best architecture:
   ```bash
   python tpot_best_pipeline.py
   ```

---

## Authors

- **Anderson Joseph Ochoa Trujillo**
- **Cristian Andres Villareal Orozco**
- **Juan Pablo Hoyos Sanchez**

---
