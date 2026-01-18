#  Pima Indian Diabetes Prediction System

A Machine Learning–powered web application that predicts whether a person is diabetic or not using medical features from the **Pima Indian Diabetes Dataset**.

This project demonstrates the complete ML workflow — from data preprocessing and model training to API deployment using **FastAPI** and frontend integration.

---

##  Key Features
- Supervised Machine Learning model using **Random Forest Classifier**
- Hyperparameter tuning for improved performance
- RESTful API built with **FastAPI**
- Model persistence using **Joblib**
- Interactive frontend for real-time predictions
- CORS enabled for frontend–backend communication

---

##  Machine Learning Details

- **Dataset:** Pima Indian Diabetes Dataset  
- **Problem Type:** Binary Classification  
- **Target Variable:**  
  - `0` → Non-Diabetic  
  - `1` → Diabetic  

- **Model Used:** Random Forest Classifier (Tuned)

###  Model Performance

Accuracy: 75.32%

Class 0 (Non-Diabetic):
Precision: 0.88
Recall: 0.72
F1-Score: 0.79

Class 1 (Diabetic):
Precision: 0.62
Recall: 0.82
F1-Score: 0.70

Overall Accuracy: 0.75



 High **recall for diabetic class (82%)**, which is especially important in medical prediction systems to reduce false negatives.


