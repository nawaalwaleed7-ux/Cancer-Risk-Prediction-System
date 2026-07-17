import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

try:
    print("Loading dataset...")

    df = pd.read_csv("data/raw/cancer_risk_dataset.csv")
    print(df.columns.tolist())
    print(df.dtypes)
    exit()

    X = df.drop(["patient_id", "cancer_risk_level"], axis=1)
    y = df["cancer_risk_level"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Preprocessor...")

    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming Test Data...")

    X_test_processed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print("Saving processed files...")

    joblib.dump(X_train_processed, "data/processed/X_train.pkl")
    joblib.dump(X_test_processed, "data/processed/X_test.pkl")
    joblib.dump(y_train, "data/processed/y_train.pkl")
    joblib.dump(y_test, "data/processed/y_test.pkl")

    print("SUCCESS!")
    print("Training Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

except Exception as e:
    print("ERROR:")
    print(e)


