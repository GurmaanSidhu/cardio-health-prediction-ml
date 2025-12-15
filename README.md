# Cardiovascular Disease Prediction using Machine Learning

This project focuses on predicting the presence of cardiovascular disease using clinical and lifestyle data. Multiple machine learning models are applied and compared to identify effective approaches for early risk assessment.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Dataset Understanding](#dataset-understanding)
5. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Model Building & Evaluation](#model-building--evaluation)
8. [Model Interpretation](#model-interpretation)
9. [Results & Conclusion](#results--conclusion)
10. [Limitations](#limitations)
11. [Future Scope](#future-scope)
12. [Technologies Used](#technologies-used)
13. [How to Run the Project](#how-to-run-the-project)

---

## Dataset

The dataset used in this project is the **Cardiovascular Disease Dataset** from Kaggle.

**Link:**  
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

The dataset contains demographic, clinical, and lifestyle information along with a binary target variable indicating the presence of cardiovascular disease.

---

## Problem Statement

Cardiovascular disease often develops gradually and may remain undetected until serious complications occur. Early identification of at-risk individuals is important for prevention and timely medical intervention.

This project aims to predict cardiovascular disease using machine learning techniques applied to healthcare-related data.

---

## Motivation

Cardiovascular risk depends on multiple interacting factors such as age, blood pressure, obesity, cholesterol, glucose levels, and lifestyle habits. Analyzing these factors together using traditional methods is challenging.

Machine learning provides an effective way to model these complex relationships and support early risk assessment.

---

## Dataset Understanding

The dataset includes numerical, ordinal, and binary features such as age, blood pressure, height, weight, cholesterol level, glucose level, and lifestyle indicators.  
Some data is self-reported, making the dataset realistic but noisy.

---

## Data Preprocessing & Feature Engineering

Key preprocessing steps:
- Converted age from days to years  
- Removed medically invalid blood pressure records  
- Handled unrealistic height and weight values  

Feature engineering:
- Body Mass Index (BMI)  
- Pulse Pressure  

These steps improved data quality and model performance.

---

## Exploratory Data Analysis (EDA)

EDA was performed to understand relationships between key risk factors and cardiovascular disease.  
Age, blood pressure, BMI, and pulse pressure showed strong associations with disease presence.

---

## Model Building & Evaluation

The following models were trained and evaluated:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  

Models were compared using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

---

## Model Interpretation

Model interpretation was performed to understand feature importance.  
Age, blood pressure, BMI, and pulse pressure consistently appeared as the most influential features across models.

---

## Results & Conclusion

Among the evaluated models, **SVM achieved the best overall performance**, showing higher accuracy and ROC-AUC.  
Logistic Regression provided a strong and interpretable baseline, while Decision Tree showed balanced performance.

Overall, models capable of capturing non-linear patterns performed better on this dataset.

---

## Limitations

This project is based on a single dataset with limited clinical and lifestyle features. Some data is self-reported and may contain inaccuracies. The dataset also represents only a single point in time.

The model is developed for academic purposes and is not intended for medical diagnosis.

---

## Future Scope

Future improvements may include using larger and more diverse datasets, adding medical test data, tracking patient data over time, and applying advanced feature engineering and model tuning techniques.

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook


---

## How to Run the Project

1. Clone the repository  
2. Install required libraries  
3. Open the Jupyter Notebook  
4. Run all cells in order  

---
