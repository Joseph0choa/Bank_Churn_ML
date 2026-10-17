# Comparative Performance Benchmark Across All Models

This document contains the consolidated benchmark results for all 15 evaluated machine learning and AutoML models for the **Bank Customer Churn Prediction** project.

---

## 1. Summary Benchmark Table (Markdown)

Evaluated on the isolated 20% holdout test set ($N = 3,000$, 20.4% minority class prevalence), alongside 5-Fold Stratified Cross-Validation on the 80% training set ($N = 12,000$) and the official Kaggle test score:

| Model | Strategy | CV ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score | G-Mean | Kaggle AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | Linear Baseline | $0.8863 \pm 0.0081$ | 0.8826 | 0.8670 | 0.7383 | 0.5392 | 0.9510 | 0.6232 | 0.7161 | 0.8871 |
| **Logistic Regression + ADASYN** | Adaptive Resampling | $0.8844 \pm 0.0088$ | 0.8822 | 0.7973 | 0.5020 | 0.8350 | 0.7877 | 0.6270 | 0.8110 | 0.8854 |
| **Random Forest** | Standard Ensemble | $0.9319 \pm 0.0044$ | 0.9320 | 0.8913 | 0.8017 | 0.6209 | 0.9606 | 0.6998 | 0.7723 | 0.9389 |
| **Extra Trees** | Standard Ensemble | $0.9319 \pm 0.0028$ | 0.9315 | 0.8870 | 0.8027 | 0.5915 | **0.9627** | 0.6811 | 0.7546 | 0.9382 |
| **Gradient Boosting + SMOTE** | Synthetic Resampling | $0.9335 \pm 0.0040$ | 0.9324 | 0.8943 | 0.7375 | 0.7484 | 0.9317 | 0.7429 | 0.8350 | 0.9399 |
| **XGBoost** | Cost-Insensitive Booster | $0.9348 \pm 0.0053$ | **0.9340** | 0.8947 | 0.7857 | 0.6650 | 0.9535 | 0.7204 | 0.7963 | **0.9416** |
| **LightGBM** | Histogram-Based Booster | $0.9325 \pm 0.0048$ | 0.9332 | **0.9007** | **0.8043** | 0.6781 | 0.9577 | 0.7358 | 0.8059 | 0.9404 |
| **LightGBM + ADASYN** | Adaptive Resampling | $0.9327 \pm 0.0038$ | 0.9323 | 0.8843 | 0.7017 | 0.7533 | 0.9179 | 0.7266 | 0.8315 | 0.9407 |
| **HistGradientBoosting** | Histogram Booster | $0.9327 \pm 0.0045$ | 0.9326 | 0.8987 | 0.7950 | 0.6781 | 0.9552 | 0.7319 | 0.8048 | 0.9413 |
| **CatBoost** | Categorical Booster | $0.9355 \pm 0.0048$ | 0.9337 | 0.8967 | 0.7949 | 0.6650 | 0.9560 | 0.7242 | 0.7974 | 0.9405 |
| **Balanced Random Forest** | Algorithmic Cost-Sensitive | $0.9317 \pm 0.0037$ | 0.9316 | 0.8693 | 0.6351 | **0.8448** | 0.8756 | 0.7251 | **0.8601** | 0.9365 |
| **Easy Ensemble** | Algorithmic Cost-Sensitive | $0.9254 \pm 0.0053$ | 0.9241 | 0.8540 | 0.6053 | 0.8170 | 0.8635 | 0.6954 | 0.8399 | 0.9304 |
| **RUSBoost** | Algorithmic Cost-Sensitive | $0.8959 \pm 0.0059$ | 0.8899 | 0.8303 | 0.5633 | 0.7484 | 0.8513 | 0.6428 | 0.7982 | 0.9082 |
| **Clean Ensemble (Top 3 Blend)** | Multi-Model Probability Average | **$0.9363 \pm 0.0045$** | **0.9343** | 0.8983 | 0.7992 | 0.6699 | 0.9569 | 0.7289 | 0.8006 | 0.9413 |
| **TPOT Pipeline (AutoML)** | Evolutionary Genetic Search | **$0.9363 \pm 0.0039$** | 0.9323 | 0.8953 | 0.7833 | 0.6732 | 0.9523 | 0.7241 | 0.8007 | 0.9406 |

---

## 2. LaTeX Table Code for Academic Manuscripts

```latex
\begin{table*}[t]
\centering
\caption{Comprehensive performance comparison across all 15 evaluated machine learning and AutoML models. Metrics evaluated on the isolated holdout test set ($N=3,000$), alongside 5-Fold Stratified Cross-Validation on the training partition ($N=12,000$) and Kaggle Public AUC. Best values per column are highlighted in bold.}
\label{tab:all_models_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccc}
\toprule
\textbf{Model} & \textbf{Imbalance / Architectural Strategy} & \textbf{CV ROC-AUC} & \textbf{Test AUC} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{Specificity} & \textbf{F1-Score} & \textbf{G-Mean} & \textbf{Kaggle AUC} \\
\midrule
Logistic Regression (Baseline) & Linear Baseline & $0.8863 \pm 0.0081$ & 0.8826 & 0.8670 & 0.7383 & 0.5392 & 0.9510 & 0.6232 & 0.7161 & 0.8871 \\
Logistic Regression + ADASYN   & Adaptive Resampling & $0.8844 \pm 0.0088$ & 0.8822 & 0.7973 & 0.5020 & 0.8350 & 0.7877 & 0.6270 & 0.8110 & 0.8854 \\
Random Forest                  & Standard Ensemble & $0.9319 \pm 0.0044$ & 0.9320 & 0.8913 & 0.8017 & 0.6209 & 0.9606 & 0.6998 & 0.7723 & 0.9389 \\
Extra Trees                    & Standard Ensemble & $0.9319 \pm 0.0028$ & 0.9315 & 0.8870 & 0.8027 & 0.5915 & \textbf{0.9627} & 0.6811 & 0.7546 & 0.9382 \\
Gradient Boosting + SMOTE      & Synthetic Resampling & $0.9335 \pm 0.0040$ & 0.9324 & 0.8943 & 0.7375 & 0.7484 & 0.9317 & 0.7429 & 0.8350 & 0.9399 \\
XGBoost                        & Cost-Insensitive Booster & $0.9348 \pm 0.0053$ & 0.9340 & 0.8947 & 0.7857 & 0.6650 & 0.9535 & 0.7204 & 0.7963 & \textbf{0.9416} \\
LightGBM                       & Histogram-Based Booster & $0.9325 \pm 0.0048$ & 0.9332 & \textbf{0.9007} & \textbf{0.8043} & 0.6781 & 0.9577 & 0.7358 & 0.8059 & 0.9404 \\
LightGBM + ADASYN              & Adaptive Resampling & $0.9327 \pm 0.0038$ & 0.9323 & 0.8843 & 0.7017 & 0.7533 & 0.9179 & 0.7266 & 0.8315 & 0.9407 \\
HistGradientBoosting           & Histogram Booster & $0.9327 \pm 0.0045$ & 0.9326 & 0.8987 & 0.7950 & 0.6781 & 0.9552 & 0.7319 & 0.8048 & 0.9413 \\
CatBoost                       & Categorical Booster & $0.9355 \pm 0.0048$ & 0.9337 & 0.8967 & 0.7949 & 0.6650 & 0.9560 & 0.7242 & 0.7974 & 0.9405 \\
Balanced Random Forest         & Algorithmic Cost-Sensitive & $0.9317 \pm 0.0037$ & 0.9316 & 0.8693 & 0.6351 & \textbf{0.8448} & 0.8756 & 0.7251 & \textbf{0.8601} & 0.9365 \\
Easy Ensemble                  & Algorithmic Cost-Sensitive & $0.9254 \pm 0.0053$ & 0.9241 & 0.8540 & 0.6053 & 0.8170 & 0.8635 & 0.6954 & 0.8399 & 0.9304 \\
RUSBoost                       & Algorithmic Cost-Sensitive & $0.8959 \pm 0.0059$ & 0.8899 & 0.8303 & 0.5633 & 0.7484 & 0.8513 & 0.6428 & 0.7982 & 0.9082 \\
Clean Ensemble (Top 3 Blend)   & Multi-Model Probability Average & \textbf{0.9363 $\pm$ 0.0045} & \textbf{0.9343} & 0.8983 & 0.7992 & 0.6699 & 0.9569 & 0.7289 & 0.8006 & 0.9413 \\
TPOT Pipeline (AutoML)         & Evolutionary Genetic Search & \textbf{0.9363 $\pm$ 0.0039} & 0.9323 & 0.8953 & 0.7833 & 0.6732 & 0.9523 & 0.7241 & 0.8007 & 0.9406 \\
\bottomrule
\end{tabular}%
}
\end{table*}
```

---

## 3. Academic Results Text (English)

Table 1 presents the comprehensive multi-metric evaluation across all 15 evaluated machine learning pipelines on the isolated holdout test set ($N = 3,000$, 20.4% minority class prevalence), alongside the internal 5-Fold Stratified Cross-Validation (CV) ROC-AUC on the training partition ($N = 12,000$) and the public Kaggle evaluation metric. The benchmark evaluates models across five architectural paradigms: (i) linear baselines, (ii) standard ensemble and gradient-boosted trees, (iii) synthetic resampling pipelines (SMOTE and ADASYN), (iv) algorithmic cost-sensitive balancing, and (v) genetic automated machine learning (AutoML via TPOT).

Across global discrimination metrics, tree-based boosting algorithms systematically outperformed linear and bagging counterparts. **XGBoost** achieved the highest discrimination on the competitive Kaggle test set ($\text{ROC-AUC} = 0.9416$) with a test ROC-AUC of $0.9340$ and balanced precision ($0.7857$). The **Top-3 Ensemble Blend** (averaging predicted posterior probabilities of CatBoost, XGBoost, and HistGradientBoosting) achieved the highest test discrimination ($\text{ROC-AUC} = 0.9343$, $\text{CV ROC-AUC} = 0.9363 \pm 0.0045$), demonstrating variance reduction through architectural diversity. In parallel, the evolutionary AutoML search (**TPOT**) converged onto a composite architecture combining a Naive Bayes stacking estimator, 64th-percentile feature selection, and gradient boosting, attaining competitive performance ($\text{Kaggle AUC} = 0.9406$, $\text{Accuracy} = 0.8953$, $\text{F1} = 0.7241$).

Crucially, the multi-metric benchmark reveals a pronounced operational trade-off between overall classification discrimination and minority churn recall. Standard gradient boosters optimized accuracy ($\text{LightGBM}: 0.9007$) and specificity ($0.9577$), yet exhibited moderate churn recall ($0.6650 - 0.6781$). Conversely, cost-sensitive and resampling techniques prioritized minority detection: **Balanced Random Forest** yielded the highest churn sensitivity ($\text{Recall} = 0.8448$, $\text{G-Mean} = 0.8601$), followed by **Logistic Regression + ADASYN** ($\text{Recall} = 0.8350$) and **Easy Ensemble** ($\text{Recall} = 0.8170$). For banking institutions prioritizing early churn intervention where the cost of customer attrition substantially exceeds that of retention campaigns, Balanced Random Forest and LightGBM + ADASYN ($\text{Recall} = 0.7533$, $\text{AUC} = 0.9323$) offer the most favorable risk-sensitive operating points.
