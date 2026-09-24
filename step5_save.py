# Step 5: train the winner (LogReg) and persist it as model.joblib.
# Also write a small model-card JSON capturing provenance.
import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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

pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)
auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

# --- Persist the model ---
joblib.dump(pipe, "model.joblib")
print(f"Saved model.joblib  (test ROC-AUC = {auc:.4f})")

# --- Persist a model card ---
card = {
    "model_name":       "telco-churn-logreg",
    "version":          "1.0",
    "created_utc":      datetime.now(timezone.utc).isoformat(),
    "algorithm":        "LogisticRegression(class_weight='balanced')",
    "features_numeric": numeric_cols,
    "features_categorical": categorical_cols,
    "target":           "Churn (Yes=1, No=0)",
    "n_train":          len(X_train),
    "n_test":           len(X_test),
    "test_roc_auc":     round(auc, 4),
    "notes": (
        "Chosen over XGBoost and RandomForest by ROC-AUC on the same split. "
        "class_weight='balanced' to counter the 73/27 class skew. "
        "Preprocessor is embedded in the pipeline - the same transform applies at serve time."
    ),
}
with open("model_card.json", "w") as f:
    json.dump(card, f, indent=2)
print("Saved model_card.json")

# --- Reload + smoke test: prove the saved artifact works ---
loaded = joblib.load("model.joblib")
sample = X_test.iloc[[0]]
p = loaded.predict_proba(sample)[0, 1]
print(f"\nReload smoke test — first test row churn probability = {p:.4f}")
print(f"Actual label for that row = {y_test.iloc[0]}")