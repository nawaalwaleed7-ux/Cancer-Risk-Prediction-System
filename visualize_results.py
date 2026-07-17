import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Create output folder
os.makedirs("outputs/figures", exist_ok=True)

# Load data
X_test = joblib.load("data/processed/X_test.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "SVM": joblib.load("models/svm.pkl")
}

accuracies = {}

# Generate confusion matrices
for name, model in models.items():

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"outputs/figures/{name.replace(' ','_')}_CM.png")
    plt.close()

# Accuracy Comparison Chart

plt.figure(figsize=(7,5))

plt.bar(
    accuracies.keys(),
    [x*100 for x in accuracies.values()]
)

plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

plt.savefig("outputs/figures/model_accuracy.png")

plt.show()

print("Graphs saved successfully!")