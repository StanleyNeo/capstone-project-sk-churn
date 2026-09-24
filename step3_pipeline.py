# Step 3: real pipeline — ColumnTransformer + balanced LogisticRegression.
# Same model as Step 2, but with all features, proper encoding, and
# class_weight='balanced' to fix the recall problem.
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

CSV = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV)

# Same cleanup as Step 2
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

# Split the column types
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print(f"Numeric features:     {len(numeric_cols)} -> {numeric_cols}")
print(f"Categorical features: {len(categorical_cols)}")

# The preprocessor: one-hot the categoricals, scale the numerics.
# handle_unknown='ignore' → safe if a category appears at serve time that
# wasn't seen at train time (returns all-zero vector, no crash).
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# The pipeline: preprocess -> classifier. One object, serialized as one file.
pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)

pred  = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("STEP 3 — Pipeline (all features, one-hot, class_weight='balanced')")
print("=" * 60)
print(f"Accuracy:            {accuracy_score(y_test, pred):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_test, proba):.4f}")
print(f"Precision (churn=1): {precision_score(y_test, pred):.4f}")
print(f"Recall (churn=1):    {recall_score(y_test, pred):.4f}")
print()
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, pred))