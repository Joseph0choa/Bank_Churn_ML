"""Model catalogue: pipeline builders and Optuna search spaces.

Every model is an ``imblearn`` Pipeline (preprocessor -> [sampler] -> classifier), so any
scaling, encoding or synthetic resampling is fitted on the training folds only.
"""
from __future__ import annotations

from catboost import CatBoostClassifier
from imblearn.ensemble import (BalancedBaggingClassifier, BalancedRandomForestClassifier,
                               EasyEnsembleClassifier, RUSBoostClassifier)
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from churn_utils import RANDOM_STATE, make_preprocessor

RS = RANDOM_STATE
N_THREADS = 2  # threads per fitted model; folds are parallelised on top of this


def _pipe(clf, sampler=None, preprocessor=None):
    steps = [("preprocessor", preprocessor if preprocessor is not None else make_preprocessor())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", clf))
    return ImbPipeline(steps)


def _split(params, prefix):
    """Separate 'prefix__x' entries (sampler/base-estimator params) from classifier params."""
    inner = {k[len(prefix) + 2:]: v for k, v in params.items() if k.startswith(prefix + "__")}
    outer = {k: v for k, v in params.items() if not k.startswith(prefix + "__")}
    return outer, inner


# --------------------------------------------------------------------------------------
# Builders: params dict -> unfitted pipeline
# --------------------------------------------------------------------------------------
def build_lr(p, preprocessor=None):
    return _pipe(LogisticRegression(max_iter=5000, random_state=RS, **p), preprocessor=preprocessor)


def build_lr_adasyn(p, preprocessor=None):
    clf, smp = _split(p, "sampler")
    return _pipe(LogisticRegression(max_iter=5000, random_state=RS, **clf),
                 ADASYN(random_state=RS, **smp), preprocessor)


def build_rf(p, preprocessor=None):
    return _pipe(RandomForestClassifier(random_state=RS, n_jobs=N_THREADS, **p), preprocessor=preprocessor)


def build_et(p, preprocessor=None):
    return _pipe(ExtraTreesClassifier(random_state=RS, n_jobs=N_THREADS, **p), preprocessor=preprocessor)


def build_gb_smote(p, preprocessor=None):
    clf, smp = _split(p, "sampler")
    return _pipe(GradientBoostingClassifier(random_state=RS, **clf), SMOTE(random_state=RS, **smp), preprocessor)


def build_xgb(p, preprocessor=None):
    return _pipe(XGBClassifier(random_state=RS, n_jobs=N_THREADS, eval_metric="logloss",
                               tree_method="hist", **p), preprocessor=preprocessor)


def build_lgbm(p, preprocessor=None):
    # subsample_freq=1 is required for LightGBM to actually apply row subsampling
    return _pipe(LGBMClassifier(random_state=RS, n_jobs=N_THREADS, subsample_freq=1, verbose=-1, **p),
                 preprocessor=preprocessor)


def build_lgbm_adasyn(p, preprocessor=None):
    clf, smp = _split(p, "sampler")
    return _pipe(LGBMClassifier(random_state=RS, n_jobs=N_THREADS, subsample_freq=1, verbose=-1, **clf),
                 ADASYN(random_state=RS, **smp), preprocessor)


def build_hgb(p, preprocessor=None):
    return _pipe(HistGradientBoostingClassifier(random_state=RS, early_stopping=False, **p),
                 preprocessor=preprocessor)


def build_catboost(p, preprocessor=None):
    return _pipe(CatBoostClassifier(random_seed=RS, thread_count=N_THREADS, verbose=0,
                                    allow_writing_files=False, **p), preprocessor=preprocessor)


def build_brf(p, preprocessor=None):
    return _pipe(BalancedRandomForestClassifier(random_state=RS, n_jobs=N_THREADS, sampling_strategy="all",
                                                replacement=True, bootstrap=False, **p),
                 preprocessor=preprocessor)


def build_easy(p, preprocessor=None):
    return _pipe(EasyEnsembleClassifier(random_state=RS, n_jobs=N_THREADS, **p), preprocessor=preprocessor)


def build_rusboost(p, preprocessor=None):
    clf, base = _split(p, "estimator")
    return _pipe(RUSBoostClassifier(estimator=DecisionTreeClassifier(random_state=RS, **base),
                                    random_state=RS, **clf), preprocessor=preprocessor)


def build_bbag(p, preprocessor=None):
    clf, base = _split(p, "estimator")
    return _pipe(BalancedBaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS, **base),
                                           random_state=RS, n_jobs=N_THREADS, **clf), preprocessor=preprocessor)


def build_svm(p, preprocessor=None):
    return _pipe(SVC(kernel="rbf", random_state=RS, cache_size=1000, **p), preprocessor=preprocessor)


def build_knn(p, preprocessor=None):
    return _pipe(KNeighborsClassifier(n_jobs=N_THREADS, **p), preprocessor=preprocessor)


HIDDEN = {"32": (32,), "64": (64,), "128": (128,), "64-32": (64, 32), "128-64": (128, 64)}


def build_mlp(p, preprocessor=None):
    p = dict(p)
    p["hidden_layer_sizes"] = HIDDEN[p.pop("hidden_layer_sizes", "64")]
    return _pipe(MLPClassifier(max_iter=500, early_stopping=True, n_iter_no_change=20,
                               random_state=RS, **p), preprocessor=preprocessor)


# --------------------------------------------------------------------------------------
# Optuna search spaces: trial -> params dict
# --------------------------------------------------------------------------------------
def _sampler_space(t, k_name):
    return {"sampler__sampling_strategy": t.suggest_float("sampler__sampling_strategy", 0.5, 1.0),
            f"sampler__{k_name}": t.suggest_int(f"sampler__{k_name}", 3, 15)}


def space_lr_adasyn(t):
    return {"C": t.suggest_float("C", 1e-3, 1e2, log=True), **_sampler_space(t, "n_neighbors")}


def _forest_space(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": t.suggest_int("max_depth", 4, 24),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
            "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8])}


def space_gb_smote(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 6),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50),
            **_sampler_space(t, "k_neighbors")}


def space_xgb(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 8),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": t.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}


def _lgbm_space(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 128, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 12),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}


def space_lgbm_adasyn(t):
    return {**_lgbm_space(t), **_sampler_space(t, "n_neighbors")}


def space_hgb(t):
    return {"max_iter": t.suggest_int("max_iter", 100, 800, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 64, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 100, log=True),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10.0, log=True)}


def space_catboost(t):
    return {"iterations": t.suggest_int("iterations", 200, 800, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": t.suggest_int("depth", 4, 8),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": t.suggest_float("random_strength", 0.1, 5.0, log=True)}


def space_brf(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": t.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
            "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8])}


def space_easy(t):
    return {"n_estimators": t.suggest_int("n_estimators", 10, 60, step=5)}


def space_rusboost(t):
    return {"n_estimators": t.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "estimator__max_depth": t.suggest_int("estimator__max_depth", 1, 6)}


def space_bbag(t):
    return {"n_estimators": t.suggest_int("n_estimators", 10, 100, step=10),
            "max_samples": t.suggest_float("max_samples", 0.5, 1.0),
            "estimator__max_depth": t.suggest_int("estimator__max_depth", 4, 24),
            "estimator__min_samples_leaf": t.suggest_int("estimator__min_samples_leaf", 1, 20)}


def space_svm(t):
    return {"C": t.suggest_float("C", 0.1, 100.0, log=True),
            "gamma": t.suggest_float("gamma", 1e-3, 1.0, log=True)}


def space_knn(t):
    return {"n_neighbors": t.suggest_int("n_neighbors", 5, 200, log=True),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
            "p": t.suggest_categorical("p", [1, 2])}


def space_mlp(t):
    return {"hidden_layer_sizes": t.suggest_categorical("hidden_layer_sizes", list(HIDDEN)),
            "alpha": t.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": t.suggest_categorical("batch_size", [64, 128, 256])}


# --------------------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------------------
# name: (builder, search space or None, n_trials, strategy family, imbalance handling)
CATALOGUE = {
    "Logistic Regression (Baseline)": (build_lr, None, 0, "Linear baseline", "None"),
    "Logistic Regression + ADASYN": (build_lr_adasyn, space_lr_adasyn, 40, "Linear", "Resampling (ADASYN)"),
    "Random Forest": (build_rf, _forest_space, 40, "Bagging ensemble", "None"),
    "Extra Trees": (build_et, _forest_space, 40, "Bagging ensemble", "None"),
    "Gradient Boosting + SMOTE": (build_gb_smote, space_gb_smote, 30, "Boosting", "Resampling (SMOTE)"),
    "XGBoost": (build_xgb, space_xgb, 50, "Boosting", "None"),
    "LightGBM": (build_lgbm, _lgbm_space, 50, "Boosting", "None"),
    "LightGBM + ADASYN": (build_lgbm_adasyn, space_lgbm_adasyn, 50, "Boosting", "Resampling (ADASYN)"),
    "HistGradientBoosting": (build_hgb, space_hgb, 50, "Boosting", "None"),
    "CatBoost": (build_catboost, space_catboost, 30, "Boosting", "None"),
    "Balanced Random Forest": (build_brf, space_brf, 40, "Bagging ensemble", "Algorithmic (balanced bootstrap)"),
    "Easy Ensemble": (build_easy, space_easy, 10, "Boosting ensemble", "Algorithmic (balanced bagging of AdaBoost)"),
    "RUSBoost": (build_rusboost, space_rusboost, 40, "Boosting", "Algorithmic (random under-sampling)"),
    "Balanced Bagging": (build_bbag, space_bbag, 40, "Bagging ensemble", "Algorithmic (balanced bagging)"),
    "SVM (RBF)": (build_svm, space_svm, 25, "Kernel method", "None"),
    "KNN": (build_knn, space_knn, 40, "Instance-based", "None"),
    "MLP": (build_mlp, space_mlp, 25, "Neural network", "None"),
}


def build(name, params, preprocessor=None):
    return CATALOGUE[name][0](params, preprocessor=preprocessor)
