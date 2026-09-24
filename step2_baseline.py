# Step 2: naive baseline — LogisticRegression on numeric-only features.
# Intentionally simple. Establishes a floor for the metric table.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

CSV = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV)

# 1) Drop customerID — 7043 unique values, would be leakage/memorization
df = df.drop(columns=["customerID"])

# 2) Coerce TotalCharges; the 11 blanks are all tenure=0 → fill with 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# 3) Target: Yes/No -> 1/0
y = (df["Churn"] == "Yes").astype(int)

# 4) Naive baseline: numeric-only features
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
X = df[numeric_cols]

# 5) Stratified split keeps the 26.54% churn ratio in both halves
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Baseline — deliberately no class_weight, no pipeline
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

pred  = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]

print("=" * 60)
print("NAIVE BASELINE — LogisticRegression, numeric-only features")
print("=" * 60)
print(f"Train size:        {len(X_train)}   Test size: {len(X_test)}")
print(f"Test churn ratio:  {y_test.mean():.4f}")
print()
print(f"Accuracy:            {accuracy_score(y_test, pred):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_test, proba):.4f}")
print(f"Precision (churn=1): {precision_score(y_test, pred):.4f}")
print(f"Recall (churn=1):    {recall_score(y_test, pred):.4f}")
print()
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, pred))