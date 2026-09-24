# Step 1: exploratory look at the Telco Churn dataset.
# Minimal on purpose — we only need shape, dtypes, class balance, and gotchas.
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

CSV = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV)

print("=" * 60)
print("SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("DTYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES (non-zero only)")
print("=" * 60)
miss = df.isna().sum()
print(miss[miss > 0] if miss.sum() else "No NaN values reported by pandas")

print("\n" + "=" * 60)
print("TARGET: Churn distribution")
print("=" * 60)
print(df["Churn"].value_counts())
print("\nRatio:")
print(df["Churn"].value_counts(normalize=True).round(4))

print("\n" + "=" * 60)
print("GOTCHA CHECK: TotalCharges dtype + coercion")
print("=" * 60)
print("dtype as loaded:", df["TotalCharges"].dtype)
coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("rows where coercion failed:", coerced.isna().sum())
print("sample of failed rows:")
print(df.loc[coerced.isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]].head())

print("\n" + "=" * 60)
print("CATEGORICAL COLUMN CARDINALITY")
print("=" * 60)
cat_cols = df.select_dtypes(include="object").columns.tolist()
for c in cat_cols:
    print(f"  {c:20s}  {df[c].nunique():3d} unique  ->  {df[c].unique()[:6].tolist()}")

print("\n" + "=" * 60)
print("NUMERIC SUMMARY")
print("=" * 60)
print(df.select_dtypes(include="number").describe().round(2))