import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df = pd.read_csv("data/raw/cancer_risk_dataset.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display statistical summary
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

plt.figure(figsize=(6,4))
sns.countplot(x='cancer_risk_level', data=df)

plt.title("Cancer Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Patients")

plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['age'], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()
plt.figure(figsize=(6,4))
plt.hist(df['bmi'], bins=20)

plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['inflammation_index'], bins=20)

plt.title("Inflammation Index Distribution")
plt.xlabel("Inflammation Index")
plt.ylabel("Frequency")

plt.show()
plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(include=['int64', 'float64'])

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()
print(df.columns)
print(df.dtypes)





