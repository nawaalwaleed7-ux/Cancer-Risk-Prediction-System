# Cancer Risk Prediction System using Machine Learning

## Project Overview

The Cancer Risk Prediction System is a machine learning-based application that predicts a patient's cancer risk level (Low, Moderate, or High) using health and medical information.

The project includes data preprocessing, model training, evaluation, visualization, and a Streamlit web application for real-time prediction.

---

## Features

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Multiple Machine Learning models
- Model comparison
- Data visualization
- Interactive Streamlit web application
- Cancer risk prediction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## Machine Learning Models

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Best Performing Model:
**Logistic Regression (80.83% Accuracy)**

---

## Project Structure

```
Cancer_Risk_Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│   └── figures/
│
├── src/
│   ├── eda.py
│   ├── preprocessor.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── visualize_results.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 80.83% |
| Random Forest | 73.33% |
| SVM | 80.00% |

---

## Future Improvements

- Hyperparameter tuning
- Larger dataset
- Deep Learning models
- Explainable AI (SHAP)
- Cloud deployment

---

## Author

**Nawal Waleed**

Bachelor's Project

Machine Learning