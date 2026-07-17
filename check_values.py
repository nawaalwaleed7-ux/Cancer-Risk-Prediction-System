import pandas as pd

df = pd.read_csv("data/raw/cancer_risk_dataset.csv")

columns = [
    "sex",
    "smoking_history",
    "alcohol_use",
    "family_history_cancer",
    "physical_activity_level",
    "diet_quality"
]

for col in columns:
    print(f"\n{col}:")
    print(df[col].unique())