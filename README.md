# Bank Customer Churn Prediction with Machine Learning and AutoML

End-to-end, leak-free benchmark of machine learning models for bank customer churn prediction: class-imbalance strategies, Bayesian hyperparameter optimisation (Optuna), genetic AutoML (TPOT), statistical model comparison and SHAP interpretability.

Data: [Bank Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge) (Kaggle). The data are synthetic; 15,000 labelled and 10,000 unlabelled customers.

---

## Methodology

| Step | Implementation |
|:--|:--|
| Split | Stratified 80/20 holdout (`random_state=42`): 12,000 train / 3,000 test, 20.4% churners in both. The test partition is used once, after all modelling decisions. |
| Leakage control | Scaling, one-hot encoding and SMOTE / ADASYN are steps of an `imblearn` `Pipeline`, refitted on the training folds of every split. |
| Hyperparameter optimisation | Optuna `TPESampler(seed=42)`, 10–50 trials per model, objective = mean ROC-AUC of stratified 5-fold CV on the training partition, no pruning. Search spaces and every trial are saved in `Results/`. |
| AutoML | TPOT 0.12.2: population 40, 15 generations, mutation 0.9 / crossover 0.1, ROC-AUC, same 5 folds, no early stopping and no global time limit. |
| Benchmark | 10 × repeated stratified 5-fold CV (50 paired folds, identical for all models): ROC-AUC, PR-AUC, Accuracy, Precision, Recall, Specificity, F1, G-Mean (mean ± std). |
| Statistics | Friedman + Nemenyi; Nadeau–Bengio corrected resampled t-test and Wilcoxon signed-rank (Holm-adjusted) vs best model and vs the logistic-regression baseline; DeLong test and bootstrap 95% CI on the holdout ROC-AUC. |
| Threshold | Metrics at 0.5 and at the G-Mean-optimal threshold chosen on out-of-fold training predictions. |
| Interpretability | Tree SHAP (summary, bar, dependence and waterfall plots) plus ranking consistency across four boosting models (Kendall's τ). |
| Ablations | Learner × imbalance strategy × ratio matrix, fidelity of synthetic samples, scalers and outlier removal, feature engineering (`TenureByAge`, …) and feature pruning. |

Models (19): logistic regression baseline and + ADASYN; Random Forest; Extra Trees; Gradient Boosting + SMOTE; XGBoost; LightGBM and + ADASYN; HistGradientBoosting; CatBoost; Balanced Random Forest; Easy Ensemble; RUSBoost; Balanced Bagging; SVM (RBF); KNN; MLP; top-3 soft-voting ensemble (equal weights, members chosen by CV); TPOT pipeline.

---

## Results

Ranked by public Kaggle AUC. Kaggle columns are the public and private leaderboard scores of the submissions generated in `Submissions/`, which are refitted on all 15,000 labelled customers. CV is mean ± std over 10 × 5-fold CV on the training partition. Test metrics are on the 3,000-customer holdout at threshold 0.5. The p-value compares each model's CV ROC-AUC with the best model by CV (Soft-Voting Ensemble (Top 3)).

| Rank | Model | Kaggle AUC | Kaggle Private AUC | CV ROC-AUC | Test ROC-AUC | Accuracy | Recall | Specificity | F1-Score | G-Mean | p vs CV-best (corr. t, Holm) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | HistGradientBoosting | 0.94192 | 0.92656 | 0.9362 ± 0.0052 | 0.9341 | 0.8980 | 0.6699 | 0.9564 | 0.7282 | 0.8005 | 0.373 |
| 2 | LightGBM + ADASYN | 0.94175 | 0.92636 | 0.9359 ± 0.0052 | 0.9349 | 0.8940 | 0.7255 | 0.9372 | 0.7363 | 0.8246 | 0.163 |
| 3 | LightGBM | 0.94165 | 0.92701 | 0.9365 ± 0.0053 | 0.9342 | 0.8983 | 0.6716 | 0.9564 | 0.7294 | 0.8014 | 0.092 |
| 4 | TPOT Pipeline | 0.94155 | 0.92555 | 0.9359 ± 0.0051 | 0.9324 | 0.8967 | 0.6520 | 0.9594 | 0.7202 | 0.7909 | 0.373 |
| 5 | Soft-Voting Ensemble (Top 3) | 0.94147 | 0.92712 | 0.9369 ± 0.0052 | 0.9350 | 0.8990 | 0.6650 | 0.9590 | 0.7287 | 0.7986 | — |
| 6 | Gradient Boosting + SMOTE | 0.94140 | 0.92626 | 0.9353 ± 0.0054 | 0.9341 | 0.8970 | 0.7271 | 0.9405 | 0.7423 | 0.8270 | 0.013 |
| 7 | CatBoost | 0.94109 | 0.92690 | 0.9368 ± 0.0054 | 0.9344 | 0.8990 | 0.6650 | 0.9590 | 0.7287 | 0.7986 | 0.510 |
| 8 | XGBoost | 0.94099 | 0.92676 | 0.9366 ± 0.0052 | 0.9353 | 0.8987 | 0.6618 | 0.9594 | 0.7271 | 0.7968 | 0.373 |
| 9 | RUSBoost | 0.93941 | 0.92454 | 0.9332 ± 0.0057 | 0.9302 | 0.8617 | 0.8448 | 0.8660 | 0.7136 | 0.8553 | 0.000 |
| 10 | Balanced Random Forest | 0.93937 | 0.92431 | 0.9326 ± 0.0060 | 0.9319 | 0.8627 | 0.8529 | 0.8652 | 0.7170 | 0.8590 | 0.001 |
| 11 | Random Forest | 0.93887 | 0.92472 | 0.9336 ± 0.0056 | 0.9327 | 0.8950 | 0.6503 | 0.9577 | 0.7165 | 0.7892 | 0.002 |
| 12 | Balanced Bagging | 0.93877 | 0.92383 | 0.9317 ± 0.0061 | 0.9319 | 0.8593 | 0.8611 | 0.8589 | 0.7141 | 0.8600 | 0.000 |
| 13 | Extra Trees | 0.93679 | 0.92325 | 0.9340 ± 0.0051 | 0.9329 | 0.8923 | 0.6193 | 0.9623 | 0.7012 | 0.7720 | 0.009 |
| 14 | MLP | 0.93442 | 0.92155 | 0.9317 ± 0.0052 | 0.9329 | 0.8927 | 0.6879 | 0.9451 | 0.7234 | 0.8063 | 0.001 |
| 15 | Easy Ensemble | 0.92985 | 0.91388 | 0.9255 ± 0.0050 | 0.9241 | 0.8540 | 0.8170 | 0.8635 | 0.6954 | 0.8399 | 0.000 |
| 16 | KNN | 0.92681 | 0.91372 | 0.9193 ± 0.0065 | 0.9245 | 0.8783 | 0.5343 | 0.9665 | 0.6418 | 0.7186 | 0.000 |
| 17 | SVM (RBF) | 0.92051 | 0.90890 | 0.9225 ± 0.0054 | 0.9201 | 0.8857 | 0.5899 | 0.9615 | 0.6779 | 0.7531 | 0.000 |
| 18 | Logistic Regression (Baseline) | 0.89368 | 0.87772 | 0.8860 ± 0.0075 | 0.8826 | 0.8670 | 0.5392 | 0.9510 | 0.6232 | 0.7161 | 0.000 |
| 19 | Logistic Regression + ADASYN | 0.89286 | 0.87835 | 0.8860 ± 0.0078 | 0.8829 | 0.8243 | 0.8056 | 0.8291 | 0.6517 | 0.8173 | 0.000 |

Key findings of the current run:

- **Kaggle:** the top 8 models are within 0.0009 public AUC of each other (0.9410–0.9419). HistGradientBoosting is first on the public leaderboard; the soft-voting ensemble is first on the private leaderboard (0.92712).
- **Statistical comparison:** the gradient-boosting family (CatBoost, XGBoost, LightGBM, HistGradientBoosting), the top-3 soft-voting ensemble, LightGBM + ADASYN and the TPOT pipeline are statistically indistinguishable in ROC-AUC (corrected resampled t-test, Holm, 5%). All tuned non-linear models are significantly better than the logistic-regression baseline (ΔAUC ≈ +0.05 in CV and on Kaggle).
- **Imbalance strategies:** resampling (SMOTE / ADASYN / RUS) and cost-sensitive learning do not improve ROC-AUC. They shift the operating point towards higher recall / G-Mean and worsen calibration (Brier score). Tuning the decision threshold on out-of-fold predictions gives the same gain in recall (≈ 0.85) without synthetic data (see `Notebooks/imbalance_and_ablation.ipynb`).
- **SHAP:** the rankings are identical across XGBoost, LightGBM, CatBoost and HistGradientBoosting (Kendall's τ = 1). The order is NumOfProducts, Age, IsActiveMember, Gender, Geography (Germany), Balance.

The full table (all metrics, LaTeX version and statistical summary) is in [`all_models_table.md`](all_models_table.md). Machine-readable versions are [`all_models_benchmark_metrics.csv`](all_models_benchmark_metrics.csv) and [`best_models_hyperparameters.json`](best_models_hyperparameters.json).

**Kaggle scores** are stored in `Results/kaggle_scores.csv`. To update them after new submissions, edit that file and re-run `train_test_models.ipynb`. Submission file numbers follow the CV ranking, so they do not change when the table is re-ranked. The previous version of the models, tables and figures is available in the git history (commit `81cc4dc`).

---

## Repository structure

```text
├── Datasets/
│   ├── train.csv                 # Raw Kaggle training file (15,000 rows)
│   ├── test.csv                  # Raw Kaggle test file (10,000 rows, unlabelled)
│   └── train_processed.csv       # train.csv without identifier columns (written by EDA.ipynb)
├── src/
│   ├── churn_utils.py            # paths, split, preprocessing, CV schemes, metrics, statistical tests
│   └── churn_models.py           # model pipelines and Optuna search spaces
├── Notebooks/
│   ├── EDA.ipynb                 # 1. data preparation, distributions, linear and non-linear dependence
│   ├── tpot_test.ipynb           # 2. TPOT evolutionary search
│   ├── train_test_models.ipynb   # 3. Optuna, benchmark, holdout, statistics, figures, SHAP, submissions, tables
│   └── imbalance_and_ablation.ipynb  # 4. imbalance matrix, synthetic-sample fidelity, scaling/outliers, feature ablation
├── Results/                      # Optuna trials/params, per-fold CV scores, statistical tests, ablation tables
├── Figs/                         # ROC curves, confusion matrices, Friedman ranks, SHAP, EDA and ablation figures
├── Submissions/                  # Kaggle submissions, one per model (not in git; generated by train_test_models.ipynb)
├── all_models_table.md           # Generated results table (Markdown + LaTeX)
├── all_models_benchmark_metrics.csv
├── best_models_hyperparameters.json
├── tpot_best_pipeline.py         # Stand-alone script for the TPOT pipeline (generated)
└── requirements.txt              # Pinned versions
```

---

## Reproduction

```bash
conda create -n churn_env python=3.11 -y
conda activate churn_env
pip install -r requirements.txt
```

Run the notebooks in this order from the `Notebooks/` folder. All paths are relative to the repository root.

1. `EDA.ipynb` (≈ 20 s)
2. `tpot_test.ipynb` (≈ 15 min on 12 cores)
3. `train_test_models.ipynb`: about 15 min of Optuna (cached in `Results/optuna_best_params.json`; set `RUN_OPTUNA_SEARCH = True` to repeat it) plus about 15 min for the benchmark
4. `imbalance_and_ablation.ipynb`

To train the TPOT pipeline on all labelled data and write a submission:

```bash
python tpot_best_pipeline.py
```

---

## Authors

- **Anderson Joseph Ochoa Trujillo**
- **Cristian Andres Villareal Orozco**
- **Juan Pablo Hoyos Sanchez**
