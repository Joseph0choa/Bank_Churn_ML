# Bank Customer Churn Prediction

Benchmark of 19 models for churn prediction on the Kaggle [Bank Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge) data, covering class-imbalance strategies, Optuna, TPOT, statistical tests and SHAP.

## Results

| Model | Kaggle public | Kaggle private | CV ROC-AUC (10×5) | Test ROC-AUC |
|:--|:--:|:--:|:--:|:--:|
| HistGradientBoosting | **0.94192** | 0.92656 | 0.9362 ± 0.0052 | 0.9341 |
| LightGBM + ADASYN | 0.94175 | 0.92636 | 0.9359 ± 0.0052 | 0.9349 |
| LightGBM | 0.94165 | 0.92701 | 0.9365 ± 0.0053 | 0.9342 |
| TPOT pipeline | 0.94155 | 0.92555 | 0.9359 ± 0.0051 | 0.9324 |
| Soft-voting ensemble (top 3) | 0.94147 | **0.92712** | **0.9369 ± 0.0052** | 0.9350 |
| Logistic regression (baseline) | 0.89368 | 0.87772 | 0.8860 ± 0.0075 | 0.8826 |

Full table (all models and metrics, with LaTeX): [`Results/benchmark/all_models_table.md`](Results/benchmark/all_models_table.md)

## Structure

```text
Datasets/     train.csv, test.csv, train_processed.csv
Notebooks/    1 EDA · 2 tpot_test · 3 train_test_models · 4 imbalance_and_ablation
src/          shared code (data, models, metrics, statistical tests, figures)
Results/      benchmark/ (tables, tests, Kaggle scores) · tuning/ (Optuna, TPOT, hyperparameters) · ablation/
Figs/         eda/ · benchmark/ · shap/ · ablation/
```

## Run

```bash
pip install -r requirements.txt
```

Run the notebooks in order from `Notebooks/`. `train_test_models.ipynb` writes the Kaggle submissions to `Submissions/`.

## Authors

Anderson Joseph Ochoa Trujillo · Cristian Andres Villareal Orozco · Juan Pablo Hoyos Sanchez
