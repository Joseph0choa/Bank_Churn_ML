# Bank Customer Churn Prediction

Benchmark of 19 models for predicting customer churn on the Kaggle [Bank Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge). The project compares gradient boosting, bagging, imbalance-aware ensembles and classical learners under the same protocol. Hyperparameters are tuned with Optuna, pipelines are searched with TPOT, model differences are checked with statistical tests, and predictions are explained with SHAP.

## Data

- `train.csv`: 15,000 customers with 10 features (credit score, geography, gender, age, tenure, balance, products, credit card, activity, salary). The target is `Exited`, with a churn rate of 20.4%.
- `test.csv`: 10,000 customers without labels, scored on the Kaggle leaderboard.
- `id`, `CustomerId` and `Surname` are dropped. `Geography` and `Gender` are one-hot encoded, and numeric features are scaled where the learner needs it.

## Method

1. **Split:** stratified 80/20 train/holdout (12,000 / 3,000), seed 42.
2. **Pipelines:** preprocessing, optional SMOTE/ADASYN and the classifier live in a single `imblearn` pipeline. Resampling therefore only touches training folds, so nothing leaks.
3. **Tuning:** Optuna TPE with a 5-fold stratified CV objective (ROC-AUC) per model. TPOT runs a genetic search (40 × 15 generations) over full pipelines.
4. **Evaluation:** 10 × 5 repeated stratified CV on the training partition (50 paired folds), followed by a single holdout test. The holdout reports ROC-AUC with a bootstrap 95% CI, plus accuracy, precision, recall, specificity, F1 and G-Mean. A G-Mean-optimal threshold is also estimated from out-of-fold predictions.
5. **Statistics:**
   - Friedman test with the Nemenyi critical difference over the 50 folds.
   - Nadeau–Bengio corrected resampled t-test and Wilcoxon test against the best model and the baseline, with Holm correction.
   - DeLong test on the holdout.
6. **Ensemble:** soft voting of the top 3 models by CV (CatBoost, XGBoost, LightGBM).
7. **Explainability:** Tree SHAP (summary, importance, dependence and a local waterfall). Kendall τ checks that the four boosters agree on the feature ranking.
8. **Kaggle:** every model is refit on all 15,000 labelled rows and submitted.

### Models

| Family | Models |
|:--|:--|
| Boosting | XGBoost, LightGBM, CatBoost, HistGradientBoosting |
| Boosting + resampling | LightGBM + ADASYN, Gradient Boosting + SMOTE |
| Bagging | Random Forest, Extra Trees |
| Imbalance-aware ensembles | Balanced Random Forest, Balanced Bagging, Easy Ensemble, RUSBoost |
| Others | MLP, SVM (RBF), KNN, Logistic Regression (baseline), Logistic Regression + ADASYN |
| AutoML / ensemble | TPOT pipeline, soft-voting ensemble (top 3) |

## Results

Top 5 models by Kaggle public AUC, plus the baseline:

| Model | Kaggle public | Kaggle private | CV ROC-AUC (10×5) | Test ROC-AUC |
|:--|:--:|:--:|:--:|:--:|
| HistGradientBoosting | **0.94192** | 0.92656 | 0.9362 ± 0.0052 | 0.9341 |
| LightGBM + ADASYN | 0.94175 | 0.92636 | 0.9359 ± 0.0052 | 0.9349 |
| LightGBM | 0.94165 | 0.92701 | 0.9365 ± 0.0053 | 0.9342 |
| TPOT pipeline | 0.94155 | 0.92555 | 0.9359 ± 0.0051 | 0.9324 |
| Soft-voting ensemble (top 3) | 0.94147 | **0.92712** | **0.9369 ± 0.0052** | 0.9350 |
| Logistic regression (baseline) | 0.89368 | 0.87772 | 0.8860 ± 0.0075 | 0.8826 |

- **The boosters form a statistically tied top group.** Friedman χ² = 801.9 (p ≈ 1e-158), Nemenyi CD = 3.96. The ensemble, CatBoost, XGBoost, LightGBM, HistGB, LightGBM + ADASYN and TPOT do not differ significantly (corrected t-test, Holm 5%).
- **Every model beats the logistic-regression baseline**, except LR + ADASYN.
- **Imbalance-aware ensembles trade AUC for recall.** Balanced Bagging, Balanced RF and RUSBoost reach recall of about 0.85 and G-Mean of about 0.86, against roughly 0.67 and 0.80 for the plain boosters, with slightly lower AUC.
- **The SHAP feature ranking is identical across XGBoost, LightGBM, CatBoost and HistGB** (Kendall τ = 1).

The full table with all 19 models and metrics, plus a LaTeX version, is in [`Results/benchmark/all_models_table.md`](Results/benchmark/all_models_table.md).

### Ablations

- **Imbalance strategy × ratio** (None, SMOTE 0.5/1.0, ADASYN 0.5/1.0) on LR, RF, XGBoost and LightGBM. Resampling raises recall and G-Mean but never improves ROC-AUC, and ADASYN 1.0 lowers it significantly for the tree models.
- **Synthetic-sample fidelity.** Both SMOTE and ADASYN produce impossible values (non-integer `Tenure`/`NumOfProducts`, fractional one-hot columns). Their samples are also distinguishable from real churners (C2ST AUC 0.76 for SMOTE, 0.82 for ADASYN).
- **Scalers.** Only the scale-sensitive learners were tested. QuantileTransformer lifts LR from 0.886 to 0.900, and StandardScaler is best for SVM.
- **Outliers.** Removing |z| > 3 rows from the training folds slightly helps LR but slightly lowers AUC for XGBoost and LightGBM.
- **Feature engineering.** `TenureByAge`, `BalanceIsZero`, `LogBalance`, `ProductsPerTenure` and feature pruning give no significant gain.

## Repository

```text
Datasets/     train.csv, test.csv, train_processed.csv
Notebooks/
  EDA.ipynb                       distributions, outliers, correlations, mutual information, Cramér's V
  tpot_test.ipynb                 TPOT pipeline search
  train_test_models.ipynb         tuning, 10×5 CV, holdout, statistical tests, ROC, SHAP, submissions
  imbalance_and_ablation.ipynb    imbalance matrix, synthetic-sample fidelity, scalers, outliers, features
src/
  churn_utils.py                  data loading, preprocessing, CV, metrics, statistical tests
  churn_models.py                 model catalogue and Optuna search spaces
  churn_report.py                 tables, ROC/confusion/Friedman figures, SHAP helpers
Results/
  benchmark/                      full metrics table, statistical tests, Kaggle scores
  tuning/                         Optuna trials and best parameters, TPOT pipeline
  ablation/                       ablation tables
Figs/         eda/ · benchmark/ · shap/ · ablation/
```

## Reproduce

Tested with Python 3.11.

```bash
conda create -n Churn python=3.11
conda activate Churn
pip install -r requirements.txt
```

Run the notebooks in order: `EDA` → `tpot_test` → `train_test_models` → `imbalance_and_ablation`.

- `train_test_models` loads the cached Optuna results by default. Set `RUN_OPTUNA_SEARCH = True` to re-run the search.
- The Kaggle submissions are written to `Submissions/`, which git does not track.
- All randomness is seeded with 42.

## Authors

Anderson Joseph Ochoa Trujillo · Cristian Andres Villareal Orozco · Juan Pablo Hoyos Sanchez
