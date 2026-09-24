# Step 4: RF + XGBoost on the same preprocessor. Pick the winner by ROC-AUC.
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from xgboost import XGBClassifier

CSV = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV)
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Same imbalance handling spirit, per model:
# - LogReg / RF: class_weight='balanced'
# - XGBoost: scale_pos_weight = negatives / positives (its equivalent)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight for XGBoost: {scale_pos_weight:.3f}")

models = {
    "LogReg (baseline)": LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric="logloss", n_jobs=-1,
    ),
}

results = []
for name, clf in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred  = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    row = {
        "model":     name,
        "accuracy":  accuracy_score(y_test, pred),
        "roc_auc":   roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall":    recall_score(y_test, pred),
    }
    results.append(row)

    print("\n" + "=" * 60)
    print(f"{name}")
    print("=" * 60)
    print(f"Accuracy:  {row['accuracy']:.4f}")
    print(f"ROC-AUC:   {row['roc_auc']:.4f}")
    print(f"Precision: {row['precision']:.4f}")
    print(f"Recall:    {row['recall']:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))

# Summary table
print("\n\n" + "=" * 78)
print("METRIC TABLE — sorted by ROC-AUC")
print("=" * 78)
res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))