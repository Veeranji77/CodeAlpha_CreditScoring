# CodeAlpha ML Internship - Task 1: Credit Scoring Model
# Goal: predict whether a person is a good or bad credit risk from past financial data.
#
# DATA NOTE: I could not download a public credit dataset in my environment,
# so make_data() builds a synthetic dataset with realistic columns
# (income, debts, payment history, etc.). If you have a real CSV
# (e.g. German Credit or "Give Me Some Credit"), pass it with:
#     python credit_scoring.py --csv your_file.csv --target default
# The target column must be 1 = defaulted / bad credit, 0 = good credit.

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------
def make_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    income = np.round(rng.lognormal(mean=10.8, sigma=0.5, size=n), 0)   # yearly income
    employment_years = np.clip(age - 21 - rng.integers(0, 10, n), 0, None)
    num_credit_cards = rng.integers(0, 8, n)
    total_debt = np.round(income * rng.uniform(0.0, 1.5, n), 0)
    credit_history_years = np.clip(rng.normal(8, 5, n), 0, None).round(1)
    late_payments = rng.poisson(1.2, n)                                  # past late payments
    loan_amount = np.round(income * rng.uniform(0.1, 1.2, n), 0)
    owns_home = rng.integers(0, 2, n)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "num_credit_cards": num_credit_cards,
        "total_debt": total_debt,
        "credit_history_years": credit_history_years,
        "late_payments": late_payments,
        "loan_amount": loan_amount,
        "owns_home": owns_home,
    })

    # probability of default depends on a mix of the columns plus noise
    score = (-2.2
             + 2.0 * (df["total_debt"] / df["income"])
             + 0.45 * df["late_payments"]
             + 0.8 * (df["loan_amount"] / df["income"])
             - 0.06 * df["credit_history_years"]
             - 0.04 * df["employment_years"]
             - 0.4 * df["owns_home"]
             + rng.normal(0, 0.8, n))
    prob = 1 / (1 + np.exp(-score))
    df["default"] = (rng.random(n) < prob).astype(int)

    # put some missing values in, like real data would have
    mask = rng.random(n) < 0.03
    df.loc[mask, "employment_years"] = np.nan
    return df


parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=None, help="optional real dataset")
parser.add_argument("--target", default="default", help="target column name")
args = parser.parse_args()

if args.csv:
    df = pd.read_csv(args.csv)
    TARGET = args.target
else:
    df = make_data()
    TARGET = "default"
    df.to_csv("credit_data_synthetic.csv", index=False)

print("Shape:", df.shape)
print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("Default rate: {:.2%}".format(df[TARGET].mean()))

# ---------------------------------------------------------------
# 2. Cleaning + feature engineering
# ---------------------------------------------------------------
# fill missing numeric values with the median
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# new features built from financial history (only when the columns exist)
if {"total_debt", "income"}.issubset(df.columns):
    df["debt_to_income"] = df["total_debt"] / df["income"]
if {"loan_amount", "income"}.issubset(df.columns):
    df["loan_to_income"] = df["loan_amount"] / df["income"]
if {"late_payments", "credit_history_years"}.issubset(df.columns):
    df["late_per_year"] = df["late_payments"] / (df["credit_history_years"] + 1)
if {"employment_years", "age"}.issubset(df.columns):
    df["employment_ratio"] = df["employment_years"] / df["age"]

# ---------------------------------------------------------------
# 3. EDA plots
# ---------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x=TARGET, data=df)
plt.title("Good (0) vs bad (1) credit")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 4. Split
# ---------------------------------------------------------------
X = df.drop(columns=TARGET)
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------------
# 5. Models
# class_weight="balanced" because defaults are the minority class
# ---------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                            random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                            class_weight="balanced", random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
fitted = {}

for name, model in models.items():
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    model.fit(X_train, y_train)
    fitted[name] = model

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "CV ROC-AUC": round(cv_auc, 4),
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "Test ROC-AUC": round(roc_auc_score(y_test, prob), 4),
    })

results_df = pd.DataFrame(results).sort_values("Test ROC-AUC", ascending=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 6. Best model details
# ---------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
best_pred = best_model.predict(X_test)
print(f"\nBest model by ROC-AUC: {best_name}")
print(classification_report(y_test, best_pred, target_names=["Good credit", "Bad credit"]))

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 5))
for name, model in fitted.items():
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=120)
plt.close()

rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5))
imp.plot(kind="barh")
plt.title("Feature importance - Random Forest")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120)
plt.close()

print("\nTop 5 important features (Random Forest):")
print(imp.sort_values(ascending=False).head(5).round(4))
print(f"\nPlots and tables saved in '{OUT_DIR}/'")
