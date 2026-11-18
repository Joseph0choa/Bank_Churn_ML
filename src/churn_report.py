"""Figures, statistical comparisons and table/JSON export used by train_test_models.ipynb."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve

from churn_utils import (FIG_DIR, N_REPEATS, N_SPLITS, RANDOM_STATE, TEST_SIZE, bootstrap_auc_ci,
                         classification_metrics, corrected_resampled_ttest, delong_test,
                         holm_correction, save_json, slugify)


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def holdout_metrics(name, y_test, score, default_thr, tuned_thr):
    """Holdout metrics at the default and at the tuned threshold, plus bootstrap CI of the AUC."""
    m_def = classification_metrics(y_test, score, (score >= default_thr).astype(int))
    m_opt = classification_metrics(y_test, score, (score >= tuned_thr).astype(int))
    lo, hi = bootstrap_auc_ci(y_test, score)
    return {**{f"Test {k}": v for k, v in m_def.items()},
            "Test ROC-AUC 95% CI": f"[{lo:.4f}, {hi:.4f}]", "Tuned threshold": tuned_thr,
            **{f"Test {k} @tuned": v for k, v in m_opt.items() if k not in ("ROC-AUC", "PR-AUC")}}


def pairwise_tests(reference, fold_scores, test_scores, holdout, y_test, order, n_train, n_val):
    """Corrected resampled t-test, Wilcoxon and DeLong of every model against ``reference`` (Holm-adjusted)."""
    rows = []
    for name in order:
        if name == reference:
            continue
        a, b = fold_scores[name].roc_auc.values, fold_scores[reference].roc_auc.values
        t, p_t = corrected_resampled_ttest(a, b, n_train, n_val)
        p_w = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        _, _, z, p_d = delong_test(y_test, test_scores[name], test_scores[reference])
        rows.append({"Model": name, "Reference": reference, "Mean ΔAUC (CV)": (a - b).mean(),
                     "t (corrected)": t, "p corrected t-test": p_t, "p Wilcoxon": p_w,
                     "ΔAUC (test)": holdout[name]["Test ROC-AUC"] - holdout[reference]["Test ROC-AUC"],
                     "z DeLong": z, "p DeLong": p_d})
    df = pd.DataFrame(rows).set_index("Model")
    for col in ["p corrected t-test", "p Wilcoxon", "p DeLong"]:
        df[col + " (Holm)"] = holm_correction(df[col].values)
    df["Significant"] = df["p corrected t-test (Holm)"] < 0.05
    return df


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def roc_figure(names, y_test, test_scores, holdout, title, path, zoom=((0.05, 0.20), (0.70, 0.90))):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    pal = sns.color_palette("tab10" if len(names) <= 10 else "tab20", len(names))
    axins = inset_axes(ax, width="45%", height="45%", loc="center right", borderpad=2)
    for c, name in zip(pal, names):
        fpr, tpr, _ = roc_curve(y_test, test_scores[name])
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {holdout[name]['Test ROC-AUC']:.4f})")
        axins.plot(fpr, tpr, color=c, lw=2)
    ax.plot([0, 1], [0, 1], ":", color="gray")
    axins.set_xlim(*zoom[0]); axins.set_ylim(*zoom[1]); axins.grid(alpha=0.3); axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.set_xlabel("False positive rate (1 − Specificity)")
    ax.set_ylabel("True positive rate (Recall)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    _clean(ax)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.show()


def _cm_heatmap(ax, cm, title):
    annot = np.array([[f"{cm[r, c]}\n({cm[r, c] / cm[r].sum():.1%})" for c in range(2)] for r in range(2)])
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=annot, fmt="", cmap="Blues", cbar=False, vmin=0, vmax=1,
                ax=ax, xticklabels=["Retained", "Churn"], yticklabels=["Retained", "Churn"], annot_kws={"fontsize": 12})
    ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")


def confusion_figures(names, y_test, test_scores, thresholds, out_dir):
    for old in out_dir.glob("*.png"):
        old.unlink()
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.8))
    for i, (name, ax) in enumerate(zip(names, axes), start=1):
        cm = confusion_matrix(y_test, (test_scores[name] >= thresholds[name]["default"]).astype(int))
        _cm_heatmap(ax, cm, name)
        f1, ax1 = plt.subplots(figsize=(4.8, 4.2))
        _cm_heatmap(ax1, cm, name)
        f1.tight_layout(); f1.savefig(out_dir / f"{i:02d}_cm_{slugify(name)}.png", bbox_inches="tight"); plt.close(f1)
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrices_top5_panel.png", bbox_inches="tight"); plt.show()


def friedman_figure(avg_rank, cd, path):
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(avg_rank) + 1))
    ypos = np.arange(len(avg_rank))[::-1]
    best = avg_rank.iloc[0]
    ax.barh(ypos, avg_rank.values, color=["#2E86AB" if r - best <= cd else "#B0B0B0" for r in avg_rank])
    ax.axvline(best + cd, color="#C73E1D", ls="--", lw=1.5, label=f"Best rank + CD ({cd:.2f})")
    ax.set_yticks(ypos); ax.set_yticklabels(avg_rank.index)
    ax.set_xlabel("Average Friedman rank of ROC-AUC (lower is better)")
    ax.set_title("Friedman ranks with Nemenyi critical difference (blue = not significantly worse than best)")
    ax.legend(loc="lower right"); _clean(ax)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.show()


def cv_boxplot(auc_matrix, order, path):
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(order) + 1))
    sns.boxplot(data=auc_matrix[order], orient="h", ax=ax, color="#9ecae1", fliersize=2)
    ax.set_xlabel("ROC-AUC (10 × 5-fold CV)"); ax.set_title("Distribution of cross-validated ROC-AUC")
    ax.grid(True, axis="x", alpha=0.3); _clean(ax)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.show()


# --------------------------------------------------------------------------------------
# SHAP helpers
# --------------------------------------------------------------------------------------
def shap_values_for(pipe, X_test):
    import shap
    pre, clf = pipe.named_steps["preprocessor"], pipe.named_steps["classifier"]
    Xt = pd.DataFrame(pre.transform(X_test), columns=pre.get_feature_names_out(), index=X_test.index)
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def display_frame(X_test, feature_names, numerical_cols):
    """Encoded feature matrix in original (unscaled) units, used for SHAP plot colours and axes."""
    return pd.DataFrame(
        np.column_stack([X_test[numerical_cols].values,
                         (X_test["Geography"] == "Germany").astype(float),
                         (X_test["Geography"] == "Spain").astype(float),
                         (X_test["Gender"] == "Male").astype(float)]),
        columns=feature_names, index=X_test.index)


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------
def _first_float(v):
    try:
        return float(str(v).split(" ")[0])
    except ValueError:
        return np.nan


def _bold_best(values):
    vals = values.map(_first_float)
    best = vals.max()
    return [f"**{v}**" if (pd.notna(x) and x == best) else str(v) for v, x in zip(values, vals)]


def export_results(final, root, *, rank_by, ensemble_members, tests_vs_best, tests_vs_baseline, best_model,
                   fried, cd, n_models, n_folds, n_train, n_test, test_rate, strategy, optuna_results,
                   search_space_doc, submission_files, tpot_summary, ensemble_name, versions):
    """Write all_models_benchmark_metrics.csv, all_models_table.md (Markdown + LaTeX) and best_models_hyperparameters.json."""
    export_cols = ["Rank", "Strategy", "CV ROC-AUC", "CV G-Mean", "Test ROC-AUC", "Test ROC-AUC 95% CI", "Test PR-AUC",
                   "Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "G-Mean", "Kaggle AUC", "Kaggle Private AUC"]
    export = final[export_cols].copy()
    num_cols = ["Test ROC-AUC", "Test PR-AUC", "Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "G-Mean",
                "Kaggle AUC", "Kaggle Private AUC"]
    for c in num_cols:
        export[c] = export[c].astype(float).round(5 if c.startswith("Kaggle") else 4)
    export.to_csv(root / "all_models_benchmark_metrics.csv")

    md_cols = ["Strategy", "CV ROC-AUC", "CV G-Mean", "Test ROC-AUC", "Test ROC-AUC 95% CI", "Accuracy", "Precision",
               "Recall", "Specificity", "F1-Score", "G-Mean", "Kaggle AUC", "Kaggle Private AUC"]
    md = export[md_cols].copy().astype(object)
    for c in md_cols[1:]:
        if c in ("Test ROC-AUC 95% CI",):
            continue
        fmt = "{:.5f}" if c.startswith("Kaggle") else "{:.4f}"
        md[c] = _bold_best(export[c].map(lambda v: fmt.format(v) if isinstance(v, float) else v).astype(str))
    md = md.replace({"nan": "—", "**nan**": "—"})

    header = "| Rank | Model | " + " | ".join(md_cols) + " |"
    sep = "|:---:|:---|" + "|".join([":---"] + [":---:"] * (len(md_cols) - 1)) + "|"
    rows = [f"| {final.loc[n, 'Rank']} | {n} | " + " | ".join(str(md.loc[n, c]) for c in md_cols) + " |" for n in md.index]

    latex_cols = ["CV ROC-AUC", "Test ROC-AUC", "Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "G-Mean",
                  "Kaggle AUC", "Kaggle Private AUC"]
    lx = [r"\begin{table*}[t]", r"\centering",
          r"\caption{Performance of all evaluated models. CV: mean $\pm$ std over 10$\times$ repeated stratified 5-fold "
          "cross-validation on the training partition ($N=" + f"{n_train:,}".replace(",", "{,}") + "$). "
          "Test: holdout partition ($N=" + f"{n_test:,}".replace(",", "{,}") + "$), "
          r"threshold 0.5. Kaggle: public and private leaderboard AUC. Best value per column in bold.}",
          r"\label{tab:all_models}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(latex_cols) + "}",
          r"\toprule", r"\textbf{Model} & " + " & ".join(rf"\textbf{{{c}}}" for c in latex_cols) + r" \\", r"\midrule"]
    for n in md.index:
        cells = [str(md.loc[n, c]).replace("±", r"$\pm$").replace("—", "--") for c in latex_cols]
        cells = [rf"\textbf{{{c.strip('*')}}}" if c.startswith("**") else c for c in cells]
        lx.append(n.replace("&", r"\&") + " & " + " & ".join(cells) + r" \\")
    lx += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]

    not_diff = ", ".join(tests_vs_best.index[~tests_vs_best["Significant"]]) or "none"
    better = ", ".join(tests_vs_baseline.index[tests_vs_baseline["Significant"] & (tests_vs_baseline["Mean ΔAUC (CV)"] > 0)]) or "none"
    text = f"""# Comparative Performance Benchmark Across All Models

*Generated automatically by `Notebooks/train_test_models.ipynb` — do not edit by hand.*

Protocol: tuned hyperparameters (Optuna, 5-fold CV on the training partition); CV columns are mean ± std over
{N_REPEATS} × repeated stratified {N_SPLITS}-fold CV ({n_folds} paired folds) on the training partition (N = {n_train:,});
test columns are computed once on the isolated holdout partition (N = {n_test:,}, {test_rate*100:.1f}% churners)
at the default decision threshold. Kaggle columns: public / private leaderboard AUC (models refitted on all labelled data).
Ranking criterion: **{rank_by}**. Ensemble members: {", ".join(ensemble_members)}.

## 1. Summary table

{header}
{sep}
{chr(10).join(rows)}

## 2. Statistical comparison (ROC-AUC)

* Friedman test over {n_models} models and {n_folds} folds: χ² = {fried.statistic:.2f}, p = {fried.pvalue:.2e};
  Nemenyi critical difference = {cd:.3f} (average ranks in `Figs/friedman_nemenyi_ranks.png`).
* Models **not** significantly different from the best model by CV ({best_model}) under the corrected resampled t-test (Holm, 5%):
  {not_diff}.
* Models significantly better than the logistic-regression baseline (corrected t-test, Holm, 5%): {better}.
* Full results: `Results/statistical_tests.csv`.

## 3. LaTeX table

```latex
{chr(10).join(lx)}
```
"""
    (root / "all_models_table.md").write_text(text)

    hyper = {"metadata": {
        "generated_by": "Notebooks/train_test_models.ipynb",
        "split": f"stratified holdout {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, random_state={RANDOM_STATE}",
        "tuning": "Optuna TPESampler(seed=42), objective = mean ROC-AUC of stratified 5-fold CV on the training partition, no pruning",
        "benchmark": f"{N_REPEATS} x repeated stratified {N_SPLITS}-fold CV on the training partition",
        "ranking_criterion": rank_by, "software": versions}, "models": {}}
    for name in final.index:
        f = final.loc[name]
        e = {"rank": int(f["Rank"]), "strategy": strategy[name], "cv_roc_auc": f["CV ROC-AUC"], "cv_g_mean": f["CV G-Mean"],
             "test_roc_auc": round(float(f["Test ROC-AUC"]), 4), "test_roc_auc_95ci": f["Test ROC-AUC 95% CI"],
             **{k.lower().replace("-", "_"): round(float(f[k]), 4)
                for k in ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "G-Mean"]},
             "tuned_threshold_oof_gmean": round(float(f["Tuned threshold"]), 4),
             "kaggle_auc": None if pd.isna(f["Kaggle AUC"]) else float(f["Kaggle AUC"]),
             "kaggle_private_auc": None if pd.isna(f["Kaggle Private AUC"]) else float(f["Kaggle Private AUC"]),
             "submission_file": f"Submissions/{submission_files[name]}"}
        if name in optuna_results:
            e.update({"optuna_trials": optuna_results[name]["n_trials"],
                      "optuna_best_tuning_cv_auc": round(optuna_results[name]["best_cv_auc"], 4),
                      "search_space": search_space_doc[name], "best_hyperparameters": optuna_results[name]["best_params"]})
        elif name == ensemble_name:
            e.update({"members": ensemble_members, "weights": "equal (soft voting)"})
        elif name == "TPOT Pipeline":
            e.update({"pipeline": tpot_summary["best_pipeline"], "tpot_config": tpot_summary["config"],
                      "tpot_search_cv_auc": round(tpot_summary["search_cv_roc_auc"], 4)})
        else:
            e.update({"best_hyperparameters": "scikit-learn defaults (max_iter=5000)"})
        hyper["models"][name] = e
    save_json(hyper, root / "best_models_hyperparameters.json")
