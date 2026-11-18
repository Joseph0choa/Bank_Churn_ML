# Benchmark results

CV: 10×5 repeated CV on train (N = 12,000). Test: holdout (N = 3,000), threshold 0.5. Ranked by Kaggle AUC.

| Rank | Model | Strategy | CV ROC-AUC | CV G-Mean | Test ROC-AUC | Test ROC-AUC 95% CI | Accuracy | Precision | Recall | Specificity | F1-Score | G-Mean | Kaggle AUC | Kaggle Private AUC |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | HistGradientBoosting | None | 0.9362 ± 0.0052 | 0.8126 ± 0.0114 | 0.9341 | [0.9233, 0.9445] | 0.8980 | 0.7977 | 0.6699 | 0.9564 | 0.7282 | 0.8005 | **0.94192** | 0.92656 |
| 2 | LightGBM + ADASYN | Resampling (ADASYN) | 0.9359 ± 0.0052 | 0.8357 ± 0.0107 | 0.9349 | [0.9242, 0.9450] | 0.8940 | 0.7475 | 0.7255 | 0.9372 | 0.7363 | 0.8246 | 0.94175 | 0.92636 |
| 3 | LightGBM | None | 0.9365 ± 0.0053 | 0.8116 ± 0.0125 | 0.9342 | [0.9233, 0.9446] | 0.8983 | 0.7981 | 0.6716 | 0.9564 | 0.7294 | 0.8014 | 0.94165 | 0.92701 |
| 4 | TPOT Pipeline | Genetic programming pipeline search | 0.9359 ± 0.0051 | 0.8121 ± 0.0124 | 0.9324 | [0.9216, 0.9427] | 0.8967 | 0.8044 | 0.6520 | 0.9594 | 0.7202 | 0.7909 | 0.94155 | 0.92555 |
| 5 | Soft-Voting Ensemble (Top 3) | Soft voting of CatBoost, XGBoost, LightGBM | **0.9369 ± 0.0052** | 0.8102 ± 0.0124 | 0.9350 | [0.9242, 0.9451] | **0.8990** | 0.8059 | 0.6650 | 0.9590 | 0.7287 | 0.7986 | 0.94147 | **0.92712** |
| 6 | Gradient Boosting + SMOTE | Resampling (SMOTE) | 0.9353 ± 0.0054 | 0.8317 ± 0.0115 | 0.9341 | [0.9232, 0.9444] | 0.8970 | 0.7581 | 0.7271 | 0.9405 | **0.7423** | 0.8270 | 0.94140 | 0.92626 |
| 7 | CatBoost | None | 0.9368 ± 0.0054 | 0.8074 ± 0.0128 | 0.9344 | [0.9236, 0.9446] | **0.8990** | 0.8059 | 0.6650 | 0.9590 | 0.7287 | 0.7986 | 0.94109 | 0.92690 |
| 8 | XGBoost | None | 0.9366 ± 0.0052 | 0.8101 ± 0.0122 | **0.9353** | [0.9247, 0.9454] | 0.8987 | 0.8068 | 0.6618 | 0.9594 | 0.7271 | 0.7968 | 0.94099 | 0.92676 |
| 9 | RUSBoost | Algorithmic (random under-sampling) | 0.9332 ± 0.0057 | **0.8571 ± 0.0074** | 0.9302 | [0.9189, 0.9409] | 0.8617 | 0.6177 | 0.8448 | 0.8660 | 0.7136 | 0.8553 | 0.93941 | 0.92454 |
| 10 | Balanced Random Forest | Algorithmic (balanced bootstrap) | 0.9326 ± 0.0060 | 0.8547 ± 0.0066 | 0.9319 | [0.9208, 0.9426] | 0.8627 | 0.6185 | 0.8529 | 0.8652 | 0.7170 | 0.8590 | 0.93937 | 0.92431 |
| 11 | Random Forest | None | 0.9336 ± 0.0056 | 0.8014 ± 0.0139 | 0.9327 | [0.9216, 0.9430] | 0.8950 | 0.7976 | 0.6503 | 0.9577 | 0.7165 | 0.7892 | 0.93887 | 0.92472 |
| 12 | Balanced Bagging | Algorithmic (balanced bagging) | 0.9317 ± 0.0061 | 0.8542 ± 0.0069 | 0.9319 | [0.9208, 0.9426] | 0.8593 | 0.6100 | **0.8611** | 0.8589 | 0.7141 | **0.8600** | 0.93877 | 0.92383 |
| 13 | Extra Trees | None | 0.9340 ± 0.0051 | 0.7838 ± 0.0120 | 0.9329 | [0.9218, 0.9431] | 0.8923 | **0.8081** | 0.6193 | 0.9623 | 0.7012 | 0.7720 | 0.93679 | 0.92325 |
| 14 | MLP | None | 0.9317 ± 0.0052 | 0.8078 ± 0.0166 | 0.9329 | [0.9220, 0.9431] | 0.8927 | 0.7627 | 0.6879 | 0.9451 | 0.7234 | 0.8063 | 0.93442 | 0.92155 |
| 15 | Easy Ensemble | Algorithmic (balanced bagging of AdaBoost) | 0.9255 ± 0.0050 | 0.8454 ± 0.0080 | 0.9241 | [0.9127, 0.9351] | 0.8540 | 0.6053 | 0.8170 | 0.8635 | 0.6954 | 0.8399 | 0.92985 | 0.91388 |
| 16 | KNN | None | 0.9193 ± 0.0065 | 0.7240 ± 0.0145 | 0.9245 | [0.9126, 0.9355] | 0.8783 | 0.8034 | 0.5343 | **0.9665** | 0.6418 | 0.7186 | 0.92681 | 0.91372 |
| 17 | SVM (RBF) | None | 0.9225 ± 0.0054 | 0.7690 ± 0.0120 | 0.9201 | [0.9076, 0.9331] | 0.8857 | 0.7969 | 0.5899 | 0.9615 | 0.6779 | 0.7531 | 0.92051 | 0.90890 |
| 18 | Logistic Regression (Baseline) | None | 0.8860 ± 0.0075 | 0.7261 ± 0.0135 | 0.8826 | [0.8663, 0.8988] | 0.8670 | 0.7383 | 0.5392 | 0.9510 | 0.6232 | 0.7161 | 0.89368 | 0.87772 |
| 19 | Logistic Regression + ADASYN | Resampling (ADASYN) | 0.8860 ± 0.0078 | 0.8164 ± 0.0085 | 0.8829 | [0.8667, 0.8991] | 0.8243 | 0.5472 | 0.8056 | 0.8291 | 0.6517 | 0.8173 | 0.89286 | 0.87835 |

- Friedman χ² = 801.9, p = 1.3e-158; Nemenyi CD = 3.96.
- Not significantly different from Soft-Voting Ensemble (Top 3) (corrected t-test, Holm 5%): CatBoost, XGBoost, LightGBM, HistGradientBoosting, LightGBM + ADASYN, TPOT Pipeline.
- Significantly better than the logistic-regression baseline: Soft-Voting Ensemble (Top 3), CatBoost, XGBoost, LightGBM, HistGradientBoosting, LightGBM + ADASYN, TPOT Pipeline, Gradient Boosting + SMOTE, Extra Trees, Random Forest, RUSBoost, Balanced Random Forest, MLP, Balanced Bagging, Easy Ensemble, SVM (RBF), KNN.

```latex
\begin{table*}[t]
\centering
\caption{Performance of all evaluated models. CV: mean $\pm$ std over 10$\times$ repeated stratified 5-fold cross-validation on the training partition ($N=12{,}000$). Test: holdout partition ($N=3{,}000$), threshold 0.5. Kaggle: public and private leaderboard AUC. Best value per column in bold.}
\label{tab:all_models}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccc}
\toprule
\textbf{Model} & \textbf{CV ROC-AUC} & \textbf{Test ROC-AUC} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{Specificity} & \textbf{F1-Score} & \textbf{G-Mean} & \textbf{Kaggle AUC} & \textbf{Kaggle Private AUC} \\
\midrule
HistGradientBoosting & 0.9362 $\pm$ 0.0052 & 0.9341 & 0.8980 & 0.7977 & 0.6699 & 0.9564 & 0.7282 & 0.8005 & \textbf{0.94192} & 0.92656 \\
LightGBM + ADASYN & 0.9359 $\pm$ 0.0052 & 0.9349 & 0.8940 & 0.7475 & 0.7255 & 0.9372 & 0.7363 & 0.8246 & 0.94175 & 0.92636 \\
LightGBM & 0.9365 $\pm$ 0.0053 & 0.9342 & 0.8983 & 0.7981 & 0.6716 & 0.9564 & 0.7294 & 0.8014 & 0.94165 & 0.92701 \\
TPOT Pipeline & 0.9359 $\pm$ 0.0051 & 0.9324 & 0.8967 & 0.8044 & 0.6520 & 0.9594 & 0.7202 & 0.7909 & 0.94155 & 0.92555 \\
Soft-Voting Ensemble (Top 3) & \textbf{0.9369 $\pm$ 0.0052} & 0.9350 & \textbf{0.8990} & 0.8059 & 0.6650 & 0.9590 & 0.7287 & 0.7986 & 0.94147 & \textbf{0.92712} \\
Gradient Boosting + SMOTE & 0.9353 $\pm$ 0.0054 & 0.9341 & 0.8970 & 0.7581 & 0.7271 & 0.9405 & \textbf{0.7423} & 0.8270 & 0.94140 & 0.92626 \\
CatBoost & 0.9368 $\pm$ 0.0054 & 0.9344 & \textbf{0.8990} & 0.8059 & 0.6650 & 0.9590 & 0.7287 & 0.7986 & 0.94109 & 0.92690 \\
XGBoost & 0.9366 $\pm$ 0.0052 & \textbf{0.9353} & 0.8987 & 0.8068 & 0.6618 & 0.9594 & 0.7271 & 0.7968 & 0.94099 & 0.92676 \\
RUSBoost & 0.9332 $\pm$ 0.0057 & 0.9302 & 0.8617 & 0.6177 & 0.8448 & 0.8660 & 0.7136 & 0.8553 & 0.93941 & 0.92454 \\
Balanced Random Forest & 0.9326 $\pm$ 0.0060 & 0.9319 & 0.8627 & 0.6185 & 0.8529 & 0.8652 & 0.7170 & 0.8590 & 0.93937 & 0.92431 \\
Random Forest & 0.9336 $\pm$ 0.0056 & 0.9327 & 0.8950 & 0.7976 & 0.6503 & 0.9577 & 0.7165 & 0.7892 & 0.93887 & 0.92472 \\
Balanced Bagging & 0.9317 $\pm$ 0.0061 & 0.9319 & 0.8593 & 0.6100 & \textbf{0.8611} & 0.8589 & 0.7141 & \textbf{0.8600} & 0.93877 & 0.92383 \\
Extra Trees & 0.9340 $\pm$ 0.0051 & 0.9329 & 0.8923 & \textbf{0.8081} & 0.6193 & 0.9623 & 0.7012 & 0.7720 & 0.93679 & 0.92325 \\
MLP & 0.9317 $\pm$ 0.0052 & 0.9329 & 0.8927 & 0.7627 & 0.6879 & 0.9451 & 0.7234 & 0.8063 & 0.93442 & 0.92155 \\
Easy Ensemble & 0.9255 $\pm$ 0.0050 & 0.9241 & 0.8540 & 0.6053 & 0.8170 & 0.8635 & 0.6954 & 0.8399 & 0.92985 & 0.91388 \\
KNN & 0.9193 $\pm$ 0.0065 & 0.9245 & 0.8783 & 0.8034 & 0.5343 & \textbf{0.9665} & 0.6418 & 0.7186 & 0.92681 & 0.91372 \\
SVM (RBF) & 0.9225 $\pm$ 0.0054 & 0.9201 & 0.8857 & 0.7969 & 0.5899 & 0.9615 & 0.6779 & 0.7531 & 0.92051 & 0.90890 \\
Logistic Regression (Baseline) & 0.8860 $\pm$ 0.0075 & 0.8826 & 0.8670 & 0.7383 & 0.5392 & 0.9510 & 0.6232 & 0.7161 & 0.89368 & 0.87772 \\
Logistic Regression + ADASYN & 0.8860 $\pm$ 0.0078 & 0.8829 & 0.8243 & 0.5472 & 0.8056 & 0.8291 & 0.6517 & 0.8173 & 0.89286 & 0.87835 \\
\bottomrule
\end{tabular}%
}
\end{table*}
```
