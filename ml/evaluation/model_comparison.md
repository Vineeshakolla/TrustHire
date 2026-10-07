# TrustHire — Candidate Model Comparison

## Performance Metrics Table (Sorted by PR-AUC)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| **XGBoost** | 0.9711 | 0.6535 | 0.8663 | 0.7450 | 0.9899 | 0.8963 |
| **Linear SVM** | 0.9838 | 0.8324 | 0.8372 | 0.8348 | 0.9834 | 0.8880 |
| **Random Forest** | 0.9782 | 0.9798 | 0.5640 | 0.7159 | 0.9890 | 0.8872 |
| **Logistic Regression** | 0.9694 | 0.6301 | 0.9012 | 0.7416 | 0.9857 | 0.8511 |
| **Multinomial Naive Bayes** | 0.9192 | 0.3664 | 0.9012 | 0.5210 | 0.9720 | 0.7909 |

## Detailed Confusion Matrix Breakdown

| Model | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| --- | --- | --- | --- | --- |
| **XGBoost** | 149 | 79 | 23 | 3278 |
| **Linear SVM** | 144 | 29 | 28 | 3328 |
| **Random Forest** | 97 | 2 | 75 | 3355 |
| **Logistic Regression** | 155 | 91 | 17 | 3266 |
| **Multinomial Naive Bayes** | 155 | 268 | 17 | 3089 |