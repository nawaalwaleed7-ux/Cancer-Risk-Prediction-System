# 🩺 Cancer Risk Prediction System

## Overview
This project is a machine learning-based web application that predicts a patient's cancer risk using health-related features. It was developed as an academic project for learning data analysis, machine learning, and web application development.

## Features
- Predicts cancer risk from patient information
- Interactive web interface using Streamlit
- Data preprocessing and model training
- Performance evaluation using multiple machine learning models

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Machine Learning Models
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
## structure
Cancer-Risk-Prediction-System
│
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   └── preprocessor.pkl
│
├── data/
│   └── cancer_risk_dataset.csv
│
├── outputs/
│   ├── Logistic_Regression_CM.png
│   ├── Random_Forest_CM.png
│   ├── SVM_CM.png
│   └── model_accuracy.png
│
└── src/
    ├── eda.py
    ├── train_model.py
    └── check_values.py

## How to Run

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Author
Nawal Waleed
BS Bioinformatics Student

