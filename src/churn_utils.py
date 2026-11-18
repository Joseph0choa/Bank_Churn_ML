"""Shared utilities for the Bank Customer Churn benchmark.

Every notebook imports from this module so that the data split, preprocessing,
cross-validation scheme, metrics and statistical tests are defined exactly once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             make_scorer, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Datasets"
SUBMISSION_DIR = PROJECT_ROOT / "Submissions"     # not tracked in git

RESULTS_DIR = PROJECT_ROOT / "Results"
BENCH_DIR = RESULTS_DIR / "benchmark"              # final tables, statistical tests, per-fold scores
TUNING_DIR = RESULTS_DIR / "tuning"                # Optuna / TPOT outputs and hyperparameters
ABLATION_DIR = RESULTS_DIR / "ablation"            # imbalance matrix and ablation studies

FIG_DIR = PROJECT_ROOT / "Figs"
FIG_EDA = FIG_DIR / "eda"
FIG_BENCH = FIG_DIR / "benchmark"
FIG_CM = FIG_BENCH / "confusion_matrices"
FIG_SHAP = FIG_DIR / "shap"
FIG_ABLATION = FIG_DIR / "ablation"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5          # folds per repetition
N_REPEATS = 10        # repetitions of stratified k-fold (10 x 5 = 50 paired estimates)

TARGET = "Exited"
ID_COLUMNS = ["id", "CustomerId", "Surname"]
CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERICAL_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                  "HasCrCard", "IsActiveMember", "EstimatedSalary"]
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

for _d in (SUBMISSION_DIR, BENCH_DIR, TUNING_DIR, ABLATION_DIR, FIG_EDA, FIG_CM, FIG_SHAP, FIG_ABLATION):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_data():
    """Return (X, y) from the processed training file (raw Kaggle train.csv without IDs)."""
    df = pd.read_csv(DATA_DIR / "train_processed.csv")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


def load_kaggle_test():
    """Return (ids, X) for the unlabelled Kaggle test file."""
    df = pd.read_csv(DATA_DIR / "test.csv")
    return df["id"], df[FEATURES].copy()


def split_data(X, y):
    """80/20 stratified holdout split shared by every notebook."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def make_preprocessor(scaler=None):
    """Scaling of numerical predictors + one-hot encoding of categorical predictors.

    It is always placed *inside* a Pipeline so it is fitted on training folds only.
    """
    return ColumnTransformer(
        transformers=[
            ("num", scaler if scaler is not None else StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
             CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_tpot_encoder():
    """One-hot encoding only (no scaling): the input representation used by the TPOT search.

    Column order equals ``pd.get_dummies(X, drop_first=True)``: the 8 numerical predictors
    followed by Geography_Germany, Geography_Spain and Gender_Male.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
             CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-motivated derived features (stateless, so it cannot leak information).

    * TenureByAge       = Tenure / Age: share of the customer's adult life spent with the bank
                          (relationship depth relative to life stage).
    * BalanceIsZero     = 1 if Balance == 0: captures the point mass at zero balance
                          (transactional / dormant accounts).
    * LogBalance        = log(1 + Balance): compresses the right-skewed balance distribution.
    * ProductsPerTenure = NumOfProducts / (Tenure + 1): product adoption intensity.
    """

    new_features = ["TenureByAge", "BalanceIsZero", "LogBalance", "ProductsPerTenure"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TenureByAge"] = X["Tenure"] / X["Age"]
        X["BalanceIsZero"] = (X["Balance"] == 0).astype(float)
        X["LogBalance"] = np.log1p(X["Balance"])
        X["ProductsPerTenure"] = X["NumOfProducts"] / (X["Tenure"] + 1.0)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else FEATURES
        return np.array(base + self.new_features)


class FEPreprocessor(BaseEstimator, TransformerMixin):
    """FeatureEngineer followed by scaling/encoding of the extended numerical set.

    Implemented as a single transformer (imblearn pipelines do not accept nested Pipelines).
    """

    def fit(self, X, y=None):
        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERICAL_COLS + FeatureEngineer.new_features),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                 CATEGORICAL_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ).fit(FeatureEngineer().transform(X))
        return self

    def transform(self, X):
        return self.ct_.transform(FeatureEngineer().transform(X))

    def get_feature_names_out(self, input_features=None):
        return self.ct_.get_feature_names_out()


def make_fe_preprocessor():
    return FEPreprocessor()


# --------------------------------------------------------------------------------------
# Cross-validation and metrics
# --------------------------------------------------------------------------------------
def tuning_cv():
    """Stratified 5-fold CV used inside Optuna objectives and TPOT."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def repeated_cv():
    """Repeated stratified 5-fold CV (10 repetitions) used for the final benchmark.

    The same splits are produced for every model, so per-fold scores are paired.
    """
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)


def _gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(sens * spec))


SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "specificity": make_scorer(recall_score, pos_label=0),
    "f1": "f1",
    "g_mean": make_scorer(_gmean),
}
METRIC_LABELS = {
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "accuracy": "Accuracy", "precision": "Precision",
    "recall": "Recall", "specificity": "Specificity", "f1": "F1-Score", "g_mean": "G-Mean",
}


def scores_from_estimator(estimator, X):
    """Continuous scores for the positive class (probability if available, else decision)."""
    if hasattr(estimator, "predict_proba"):
        try:
            return estimator.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return estimator.decision_function(X)


def classification_metrics(y_true, y_score, y_pred):
    """All holdout metrics for one model."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "PR-AUC": average_precision_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": rec,
        "Specificity": spec,
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "G-Mean": float(np.sqrt(rec * spec)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def gmean_optimal_threshold(y_true, y_score):
    """Threshold that maximises G-Mean on (out-of-fold) training predictions."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    g = np.sqrt(tpr * (1 - fpr))
    i = int(np.argmax(g[1:]) + 1)  # skip the +inf threshold
    return float(thr[i])


# --------------------------------------------------------------------------------------
# Statistical tests
# --------------------------------------------------------------------------------------
def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV.

    a, b: paired per-fold scores of two models obtained on identical folds.
    Returns (t statistic, two-sided p-value).
    """
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / n + n_test / n_train) * var)
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    return float(t), float(p)


def holm_correction(pvalues):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def nemenyi_cd(n_models, n_datasets, alpha=0.05):
    """Critical difference of the Nemenyi post-hoc test (Demsar, 2006)."""
    q = stats.studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets)))


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y_true, score_a, score_b):
    """DeLong et al. (1988) test for two correlated ROC-AUCs on the same test set.

    Returns (auc_a, auc_b, z, two-sided p-value). Implementation follows Sun & Xu (2014).
    """
    y_true = np.asarray(y_true)
    pos = y_true == 1
    m, n = pos.sum(), (~pos).sum()
    aucs, v01, v10 = [], [], []
    for s in (np.asarray(score_a, float), np.asarray(score_b, float)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[~pos]), _midrank(s)
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v01.append((tz[pos] - tx) / n)
        v10.append(1.0 - (tz[~pos] - ty) / m)
    sx, sy = np.cov(np.vstack(v01)), np.cov(np.vstack(v10))
    s = sx / m + sy / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, alpha=0.05, seed=RANDOM_STATE):
    """Stratified percentile bootstrap confidence interval of the holdout ROC-AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs[b] = roc_auc_score(y_true[idx], y_score[idx])
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------
def slugify(name):
    keep = "".join(c.lower() if c.isalnum() else "_" for c in name)
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")


def save_json(obj, path):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, default=_default))


def load_json(path):
    return json.loads(Path(path).read_text())
