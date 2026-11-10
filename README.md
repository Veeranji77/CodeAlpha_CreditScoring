# CodeAlpha_CreditScoring

Machine Learning Internship at CodeAlpha - **Task 1: Credit Scoring Model**

## Objective
Predict whether a person is likely to default (bad credit) using past financial data.

## Dataset
The script generates a **synthetic** dataset of 5000 people (columns: age, income, employment years, credit cards, total debt, credit history length, late payments, loan amount, owns home, default). About 31% are defaults, and about 3% of employment years are missing on purpose to practise cleaning.

Because it is synthetic, the scores below show that the pipeline works, but they do not say how a model would do on real bank data. To use a real dataset (for example German Credit or Give Me Some Credit):
```
python credit_scoring.py --csv your_file.csv --target default
```
The target column should be 1 = default / bad credit and 0 = good credit.

## Approach
1. Filled missing values with the median
2. Feature engineering: `debt_to_income`, `loan_to_income`, `late_per_year`, `employment_ratio`
3. 80/20 stratified split
4. Trained Logistic Regression, Decision Tree and Random Forest (`class_weight="balanced"` since defaults are the minority)
5. Evaluated with Precision, Recall, F1 and ROC-AUC, plus 5-fold cross-validation

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.698 | 0.5123 | 0.6667 | 0.5794 | 0.7690 |
| Random Forest | 0.700 | 0.5167 | 0.5962 | 0.5536 | 0.7463 |
| Decision Tree | 0.659 | 0.4638 | 0.5962 | 0.5217 | 0.7139 |

Logistic Regression did best on ROC-AUC and F1. The data has a lot of random noise built in, so the scores are moderate and the simple linear model beat the tree models.
Most important features: `debt_to_income`, `total_debt`, `late_per_year`.

## How to run
```
pip install -r requirements.txt
python credit_scoring.py
```
Plots and the results table are saved in the `outputs/` folder.

## Files
- `credit_scoring.py` - full code
- `credit_data_synthetic.csv` - generated data
- `outputs/` - confusion matrix, ROC curves, feature importance, correlation heatmap
- `requirements.txt`
