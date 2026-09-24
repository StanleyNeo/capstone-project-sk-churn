@"
# Data

## Source
IBM Telco Customer Churn
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## File
WA_Fn-UseC_-Telco-Customer-Churn.csv
~7043 rows x 21 columns

## Not committed
The CSV lives at data/raw/ and is gitignored. To reproduce:
1. Download from the Kaggle link above
2. Place the CSV at data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
"@ | Out-File -Encoding utf8 data\README.md