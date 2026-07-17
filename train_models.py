import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# ==========================
# Load Data
# ==========================

X_train = joblib.load("data/processed/X_train.pkl")
X_test = joblib.load("data/processed/X_test.pkl")
y_train = joblib.load("data/processed/y_train.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

print("Dataset Loaded Successfully!\n")

# ==========================
# Logistic Regression
# ==========================

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_pred)

print("Logistic Regression Accuracy:", round(logistic_accuracy*100,2), "%")

# ==========================
# Random Forest
# ==========================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", round(rf_accuracy*100,2), "%")

# ==========================
# Support Vector Machine
# ==========================

svm_model = SVC()

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", round(svm_accuracy*100,2), "%")

# ==========================
# Find Best Model
# ==========================

accuracies = {
    "Logistic Regression": logistic_accuracy,
    "Random Forest": rf_accuracy,
    "SVM": svm_accuracy
}

best_model_name = max(accuracies, key=accuracies.get)

print("\nBest Model:", best_model_name)

# ==========================
# Save Models
# ==========================

import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

joblib.dump(logistic_model, "models/logistic_regression.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(svm_model, "models/svm.pkl")

print("Models saved successfully!")
print(os.listdir("models"))