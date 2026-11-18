"""Figures, statistical comparisons and table/JSON export used by train_test_models.ipynb."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve

from churn_utils import (BENCH_DIR, TUNING_DIR, N_REPEATS, N_SPLITS, RANDOM_STATE, TEST_SIZE, bootstrap_auc_ci,
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


ROC_COLORS = ["#d62728", "#1f77b4", "#009e73", "#e7298a", "#5e3c99", "#0b1f4b", "#66bd63", "#e69f00", "#56b4e9"]
SHORT_NAMES = {"Soft-Voting Ensemble (Top 3)": "Ensemble (Top 3)", "Gradient Boosting + SMOTE": "GB + SMOTE",
               "HistGradientBoosting": "HistGB", "TPOT Pipeline": "TPOT", "Logistic Regression (Baseline)": "Logistic Regression",
               "Logistic Regression + ADASYN": "LR + ADASYN", "Balanced Random Forest": "Balanced RF"}
ROC_STYLES = ["-", (0, (5, 2)), "-.", (0, (1, 2)), (0, (6, 3, 1, 3)), (0, (8, 2, 2, 2)), (0, (1, 1)),
              (0, (3, 1, 1, 1)), (0, (4, 4))]


def roc_figure(names, y_test, test_scores, path, colors=None):
    """ROC curves in the manuscript style: zoomed inset of FPR [0, 0.2] x TPR [0.8, 1.0]."""
    from sklearn.metrics import roc_auc_score
    names = sorted(names, key=lambda n: roc_auc_score(y_test, test_scores[n]), reverse=True)
    colors = colors or ROC_COLORS
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(10, 8.5))
    axins = ax.inset_axes([0.30, 0.42, 0.38, 0.32])
    for i, name in enumerate(names):
        fpr, tpr, _ = roc_curve(y_test, test_scores[name])
        auc_v = roc_auc_score(y_test, test_scores[name])
        kw = dict(color=colors[i % len(colors)], ls=ROC_STYLES[i % len(ROC_STYLES)],
                  lw=3.5 if i == 0 else 2.2, zorder=10 - i)
        label = SHORT_NAMES.get(name, name)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc_v:.3f})", **kw)
        axins.plot(fpr, tpr, **kw)
    for a_ in (ax, axins):
        a_.plot([0, 1], [0, 1], ls=":", color="#3a3ab0", lw=1.5, zorder=0)
        a_.grid(True, color="#d9d9d9", lw=0.8)
        a_.set_axisbelow(True)
    ax.set_xlim(-0.005, 1.0); ax.set_ylim(0.0, 1.01)
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("False Positive Rate", fontsize=18); ax.set_ylabel("True Positive Rate", fontsize=18)
    ax.tick_params(labelsize=16)
    axins.set_xlim(0.0, 0.2); axins.set_ylim(0.8, 1.0)
    axins.set_xticks([0.0, 0.1, 0.2]); axins.set_yticks([0.8, 0.9, 1.0])
    axins.tick_params(labelsize=12)
    axins.set_facecolor("white")
    # zoom rectangle with connectors from its lower-left and upper-right corners
    rect, connectors = ax.indicate_inset_zoom(axins, edgecolor="#8c8c8c", alpha=1.0, lw=1.2)
    for c, visible in zip(connectors, [True, False, False, True]):  # lower-left, upper-left, lower-right, upper-right
        c.set_visible(visible)
        c.set_color("#8c8c8c"); c.set_linewidth(1.2)
    ax.legend(loc="lower right", fontsize=14, frameon=True, framealpha=0.95, edgecolor="#d9d9d9",
              handlelength=2.6, borderpad=0.6)
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.show()


def _cm_heatmap(ax, cm, title):
    """Confusion matrix in the manuscript style: counts, Blues colour scale with colour bar."""
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax, annot_kws={"fontsize": 14},
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.tick_params(labelsize=12)


def confusion_figures(names, y_test, test_scores, thresholds, out_dir):
    """One file per model plus a 3 + 2 panel (top row 3, bottom row 2 centred)."""
    for old in out_dir.glob("*.png"):
        old.unlink()
    fig = plt.figure(figsize=(21, 12))
    gs = fig.add_gridspec(2, 6)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    for i, (name, slot) in enumerate(zip(names, slots), start=1):
        cm = confusion_matrix(y_test, (test_scores[name] >= thresholds[name]["default"]).astype(int))
        label = SHORT_NAMES.get(name, name)
        _cm_heatmap(fig.add_subplot(slot), cm, label)
        f1, ax1 = plt.subplots(figsize=(7, 5.8))
        _cm_heatmap(ax1, cm, label)
        f1.tight_layout(); f1.savefig(out_dir / f"{i:02d}_cm_{slugify(name)}.png", dpi=300, bbox_inches="tight")
        plt.close(f1)
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrices_top5_panel.png", dpi=300, bbox_inches="tight")
    plt.show()


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


def export_results(final, *, rank_by, ensemble_members, tests_vs_best, tests_vs_baseline, best_model,
                   fried, cd, n_models, n_folds, n_train, n_test, test_rate, strategy, optuna_results,
                   search_space_doc, submission_files, tpot_summary, ensemble_name, versions):
    """Write Results/benchmark/all_models_benchmark_metrics.csv, all_models_table.md and Results/tuning/best_models_hyperparameters.json."""
    export_cols = ["Rank", "Strategy", "CV ROC-AUC", "CV G-Mean", "Test ROC-AUC", "Test ROC-AUC 95% CI", "Test PR-AUC",
                   "Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "G-Mean", "Kaggle AUC", "Kaggle Private AUC"]
    export = final[export_cols].copy()
    num_cols = ["Test ROC-AUC", "Test PR-AUC", "Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "G-Mean",
                "Kaggle AUC", "Kaggle Private AUC"]
    for c in num_cols:
        export[c] = export[c].astype(float).round(5 if c.startswith("Kaggle") else 4)
    export.to_csv(BENCH_DIR / "all_models_benchmark_metrics.csv")

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
    text = f"""# Benchmark results

CV: 10×5 repeated CV on train (N = {n_train:,}). Test: holdout (N = {n_test:,}), threshold 0.5. Ranked by {rank_by}.

{header}
{sep}
{chr(10).join(rows)}

- Friedman χ² = {fried.statistic:.1f}, p = {fried.pvalue:.1e}; Nemenyi CD = {cd:.2f}.
- Not significantly different from {best_model} (corrected t-test, Holm 5%): {not_diff}.
- Significantly better than the logistic-regression baseline: {better}.

```latex
{chr(10).join(lx)}
```
"""
    (BENCH_DIR / "all_models_table.md").write_text(text)

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
    save_json(hyper, TUNING_DIR / "best_models_hyperparameters.json")
